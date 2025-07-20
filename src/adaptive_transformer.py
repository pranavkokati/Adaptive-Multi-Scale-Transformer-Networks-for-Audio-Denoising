import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from .adaptive_scaling import AdaptiveComputationAllocation
from .contrastive_learning import SelfSupervisedContrastiveLearning
from .cross_attention import ProgressiveCrossModalAttention
from .multi_scale_noise import MultiScaleNoiseCharacterization
from .diffusion_enhancer import DiffusionEnhancer
from .hybrid_denoiser import HybridDenoiser
from .multitask_pipeline import MultiTaskEnhancer
from .lightweight_model import LightweightEnhancer
from .pretrained_audio import PretrainedFeatureExtractor, PretrainedVocoder


class SpectrogramProcessor(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        self.stft = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=None
        )
        
        self.istft = T.InverseSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )
        
    def forward(self, audio):
        spec = self.stft(audio)
        magnitude = torch.abs(spec)
        phase = torch.angle(spec)
        return magnitude, phase
    
    def inverse(self, magnitude, phase):
        complex_spec = magnitude * torch.exp(1j * phase)
        audio = self.istft(complex_spec)
        return audio


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.time_encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 4, 7, padding=3),
            nn.GroupNorm(8, hidden_dim // 4),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, 5, padding=2),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU()
        )
        
        self.freq_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU()
        )
        
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 1000, hidden_dim) * 0.02
        )
        
    def forward(self, time_signal, freq_magnitude):
        batch_size, length = time_signal.shape
        
        time_features = self.time_encoder(time_signal.unsqueeze(1))
        time_features = time_features.transpose(1, 2)
        
        freq_features = self.freq_encoder(freq_magnitude.transpose(1, 2))
        
        seq_len = min(time_features.shape[1], freq_features.shape[1])
        time_features = time_features[:, :seq_len]
        freq_features = freq_features[:, :seq_len]
        
        if seq_len <= self.positional_encoding.shape[1]:
            pos_enc = self.positional_encoding[:, :seq_len]
            time_features = time_features + pos_enc
            freq_features = freq_features + pos_enc
        
        return time_features, freq_features


class OutputDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.magnitude_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
        
        self.phase_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()
        )
        
        self.time_decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim // 2, hidden_dim // 4, 5, padding=2),
            nn.GroupNorm(8, hidden_dim // 4),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim // 4, 1, 7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, features, original_magnitude, original_phase):
        enhanced_magnitude = self.magnitude_decoder(features)
        enhanced_magnitude = enhanced_magnitude * original_magnitude
        
        phase_residual = self.phase_decoder(features) * 0.1
        enhanced_phase = original_phase + phase_residual
        
        time_features = features.transpose(1, 2)
        time_residual = self.time_decoder(time_features).squeeze(1)
        
        return enhanced_magnitude, enhanced_phase, time_residual


class AdaptiveMultiScaleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.spectrogram_processor = SpectrogramProcessor(
            n_fft=config.data.n_fft,
            hop_length=config.data.hop_length,
            win_length=config.data.win_length
        )
        
        self.noise_characterization = MultiScaleNoiseCharacterization(
            input_channels=1,
            hidden_dim=config.model.hidden_dim,
            num_scales=config.model.num_scales
        )
        
        freq_dim = config.data.n_fft // 2 + 1
        self.feature_encoder = FeatureEncoder(freq_dim, config.model.hidden_dim)
        
        self.cross_modal_attention = ProgressiveCrossModalAttention(
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.num_heads,
            num_layers=config.model.base_layers,
            dropout=config.model.dropout
        )
        
        self.adaptive_computation = AdaptiveComputationAllocation(
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.num_heads,
            base_layers=config.model.base_layers,
            max_layers=config.model.max_layers,
            dropout=config.model.dropout
        )
        
        self.output_decoder = OutputDecoder(config.model.hidden_dim, freq_dim)
        
        self.contrastive_learning = None
        
        # Initialize alternative enhancement methods
        self.diffusion_enhancer = DiffusionEnhancer()
        self.hybrid_denoiser = HybridDenoiser()
        self.multitask_enhancer = MultiTaskEnhancer()
        self.lightweight_enhancer = LightweightEnhancer()
        self.pretrained_extractor = PretrainedFeatureExtractor()
        
        # Enhancement method selection
        self.enhancement_method = getattr(config, 'enhancement_method', 'transformer')
        
    def setup_contrastive_learning(self):
        encoder = nn.Sequential(
            self.feature_encoder,
            self.cross_modal_attention
        )
        self.contrastive_learning = SelfSupervisedContrastiveLearning(
            encoder=encoder,
            hidden_dim=self.config.model.hidden_dim,
            temperature=self.config.model.temperature
        )
    
    def forward(self, noisy_audio, clean_audio=None, mode='inference'):
        if self.enhancement_method == 'diffusion':
            return self.diffusion_enhancer(noisy_audio)
        elif self.enhancement_method == 'hybrid':
            return self.hybrid_denoiser.process(noisy_audio)
        elif self.enhancement_method == 'multitask':
            return self.multitask_enhancer.process(noisy_audio)
        elif self.enhancement_method == 'lightweight':
            return self.lightweight_enhancer(noisy_audio)
        else:
            return self._transformer_forward(noisy_audio, mode)
    
    def _transformer_forward(self, noisy_audio, mode):
        batch_size, length = noisy_audio.shape
        
        noise_analysis = self.noise_characterization(noisy_audio.unsqueeze(1))
        
        magnitude, phase = self.spectrogram_processor(noisy_audio)
        time_features, freq_features = self.feature_encoder(noisy_audio, magnitude)
        
        cross_attention_output = self.cross_modal_attention(time_features, freq_features)
        
        adaptive_output = self.adaptive_computation(
            cross_attention_output['output'],
            noise_analysis['complexity_scores']
        )
        
        enhanced_magnitude, enhanced_phase, time_residual = self.output_decoder(
            adaptive_output['output'], magnitude, phase
        )
        
        enhanced_audio = self.spectrogram_processor.inverse(enhanced_magnitude, enhanced_phase)
        enhanced_audio = enhanced_audio + time_residual[:, :enhanced_audio.shape[1]]
        
        output = {
            'enhanced_audio': enhanced_audio,
            'enhanced_magnitude': enhanced_magnitude,
            'enhanced_phase': enhanced_phase,
            'noise_analysis': noise_analysis,
            'cross_attention_output': cross_attention_output,
            'adaptive_output': adaptive_output
        }
        
        if mode == 'training' and clean_audio is not None and self.contrastive_learning is not None:
            contrastive_output = self.contrastive_learning(clean_audio, noisy_audio)
            output['contrastive_output'] = contrastive_output
        
        return output
    
    def compute_loss(self, output, target_audio, noisy_audio):
        enhanced_audio = output['enhanced_audio']
        
        reconstruction_loss = F.mse_loss(enhanced_audio, target_audio)
        
        magnitude_loss = F.l1_loss(
            output['enhanced_magnitude'],
            torch.abs(self.spectrogram_processor(target_audio)[0])
        )
        
        spectral_loss = 0.5 * reconstruction_loss + 0.3 * magnitude_loss
        
        total_loss = spectral_loss
        
        if 'contrastive_output' in output:
            contrastive_loss = output['contrastive_output']['contrastive_loss']
            total_loss = total_loss + 0.2 * contrastive_loss
        
        efficiency_penalty = torch.mean(
            torch.relu(output['adaptive_output']['predicted_latency'] - 0.025)
        )
        total_loss = total_loss + 0.1 * efficiency_penalty
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'magnitude_loss': magnitude_loss,
            'spectral_loss': spectral_loss,
            'contrastive_loss': output.get('contrastive_output', {}).get('contrastive_loss', torch.tensor(0.0)),
            'efficiency_penalty': efficiency_penalty
        }
