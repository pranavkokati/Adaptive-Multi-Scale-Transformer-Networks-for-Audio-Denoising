import torch
import numpy as np
from pesq import pesq
from pystoi import stoi
import librosa
from scipy import signal


class AudioMetrics:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def pesq_score(self, reference, enhanced):
        reference_np = reference.detach().cpu().numpy()
        enhanced_np = enhanced.detach().cpu().numpy()
        
        if reference_np.ndim > 1:
            reference_np = reference_np.squeeze()
        if enhanced_np.ndim > 1:
            enhanced_np = enhanced_np.squeeze()
        
        try:
            score = pesq(self.sample_rate, reference_np, enhanced_np, 'wb')
            return score
        except:
            return 0.0
    
    def stoi_score(self, reference, enhanced):
        reference_np = reference.detach().cpu().numpy()
        enhanced_np = enhanced.detach().cpu().numpy()
        
        if reference_np.ndim > 1:
            reference_np = reference_np.squeeze()
        if enhanced_np.ndim > 1:
            enhanced_np = enhanced_np.squeeze()
        
        try:
            score = stoi(reference_np, enhanced_np, self.sample_rate, extended=False)
            return score
        except:
            return 0.0
    
    def si_sdr(self, reference, enhanced):
        reference = reference.view(-1)
        enhanced = enhanced.view(-1)
        
        alpha = torch.sum(enhanced * reference) / torch.sum(reference ** 2)
        target = alpha * reference
        noise = enhanced - target
        
        si_sdr_value = 10 * torch.log10(torch.sum(target ** 2) / torch.sum(noise ** 2))
        return si_sdr_value.item()
    
    def snr(self, reference, enhanced):
        reference = reference.view(-1)
        enhanced = enhanced.view(-1)
        
        noise = enhanced - reference
        signal_power = torch.mean(reference ** 2)
        noise_power = torch.mean(noise ** 2)
        
        snr_value = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        return snr_value.item()
    
    def spectral_distortion(self, reference, enhanced):
        ref_spec = torch.stft(reference, n_fft=1024, hop_length=256, return_complex=True)
        enh_spec = torch.stft(enhanced, n_fft=1024, hop_length=256, return_complex=True)
        
        ref_mag = torch.abs(ref_spec)
        enh_mag = torch.abs(enh_spec)
        
        log_ref = torch.log(ref_mag + 1e-8)
        log_enh = torch.log(enh_mag + 1e-8)
        
        spectral_dist = torch.mean((log_ref - log_enh) ** 2)
        return spectral_dist.item()
    
    def compute_all_metrics(self, reference, enhanced):
        metrics = {}
        
        if reference.dim() > 1:
            batch_pesq = []
            batch_stoi = []
            batch_si_sdr = []
            batch_snr = []
            batch_spectral = []
            
            for i in range(reference.shape[0]):
                batch_pesq.append(self.pesq_score(reference[i], enhanced[i]))
                batch_stoi.append(self.stoi_score(reference[i], enhanced[i]))
                batch_si_sdr.append(self.si_sdr(reference[i], enhanced[i]))
                batch_snr.append(self.snr(reference[i], enhanced[i]))
                batch_spectral.append(self.spectral_distortion(reference[i], enhanced[i]))
            
            metrics['pesq'] = np.mean(batch_pesq)
            metrics['stoi'] = np.mean(batch_stoi)
            metrics['si_sdr'] = np.mean(batch_si_sdr)
            metrics['snr'] = np.mean(batch_snr)
            metrics['spectral_distortion'] = np.mean(batch_spectral)
        else:
            metrics['pesq'] = self.pesq_score(reference, enhanced)
            metrics['stoi'] = self.stoi_score(reference, enhanced)
            metrics['si_sdr'] = self.si_sdr(reference, enhanced)
            metrics['snr'] = self.snr(reference, enhanced)
            metrics['spectral_distortion'] = self.spectral_distortion(reference, enhanced)
        
        return metrics


class PerceptualMetrics:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def bark_spectral_distortion(self, reference, enhanced):
        def hz_to_bark(f):
            return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)
        
        ref_np = reference.detach().cpu().numpy()
        enh_np = enhanced.detach().cpu().numpy()
        
        f, t, ref_spec = signal.spectrogram(ref_np, self.sample_rate, nperseg=1024)
        _, _, enh_spec = signal.spectrogram(enh_np, self.sample_rate, nperseg=1024)
        
        bark_freqs = hz_to_bark(f)
        
        ref_bark = np.zeros((25, ref_spec.shape[1]))
        enh_bark = np.zeros((25, enh_spec.shape[1]))
        
        for i in range(25):
            bark_start = i
            bark_end = i + 1
            freq_mask = (bark_freqs >= bark_start) & (bark_freqs < bark_end)
            
            if np.any(freq_mask):
                ref_bark[i] = np.mean(ref_spec[freq_mask], axis=0)
                enh_bark[i] = np.mean(enh_spec[freq_mask], axis=0)
        
        log_ref = np.log(ref_bark + 1e-8)
        log_enh = np.log(enh_bark + 1e-8)
        
        bsd = np.mean((log_ref - log_enh) ** 2)
        return bsd
    
    def segmental_snr(self, reference, enhanced, frame_length=1024):
        ref_np = reference.detach().cpu().numpy()
        enh_np = enhanced.detach().cpu().numpy()
        
        num_frames = len(ref_np) // frame_length
        snr_values = []
        
        for i in range(num_frames):
            start = i * frame_length
            end = start + frame_length
            
            ref_frame = ref_np[start:end]
            enh_frame = enh_np[start:end]
            
            signal_power = np.mean(ref_frame ** 2)
            noise_power = np.mean((enh_frame - ref_frame) ** 2)
            
            if noise_power > 0 and signal_power > 0:
                frame_snr = 10 * np.log10(signal_power / noise_power)
                snr_values.append(frame_snr)
        
        return np.mean(snr_values) if snr_values else 0.0


class ComputationalMetrics:
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start_timer(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time:
            self.start_time.record()
    
    def end_timer(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.end_time:
            self.end_time.record()
    
    def get_latency(self):
        if self.start_time and self.end_time:
            torch.cuda.synchronize()
            return self.start_time.elapsed_time(self.end_time) / 1000.0
        return 0.0
    
    def memory_usage(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    
    def model_size(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb


class EvaluationSuite:
    def __init__(self, sample_rate=16000):
        self.audio_metrics = AudioMetrics(sample_rate)
        self.perceptual_metrics = PerceptualMetrics(sample_rate)
        self.computational_metrics = ComputationalMetrics()
    
    def evaluate_batch(self, model, batch, device='cpu'):
        model.eval()
        
        clean_audio = batch['clean'].to(device)
        noisy_audio = batch['noisy'].to(device)
        
        self.computational_metrics.start_timer()
        
        with torch.no_grad():
            output = model(noisy_audio, mode='inference')
            enhanced_audio = output['enhanced_audio']
        
        self.computational_metrics.end_timer()
        
        results = {}
        
        audio_metrics = self.audio_metrics.compute_all_metrics(clean_audio, enhanced_audio)
        results.update(audio_metrics)
        
        results['latency'] = self.computational_metrics.get_latency()
        results['memory_usage'] = self.computational_metrics.memory_usage()
        
        if clean_audio.shape[0] == 1:
            results['bark_spectral_distortion'] = self.perceptual_metrics.bark_spectral_distortion(
                clean_audio[0], enhanced_audio[0]
            )
            results['segmental_snr'] = self.perceptual_metrics.segmental_snr(
                clean_audio[0], enhanced_audio[0]
            )
        
        return results
    
    def evaluate_dataset(self, model, dataloader, device='cpu', max_batches=None):
        model.eval()
        
        all_results = []
        
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            batch_results = self.evaluate_batch(model, batch, device)
            all_results.append(batch_results)
        
        aggregated_results = {}
        for key in all_results[0].keys():
            values = [result[key] for result in all_results if not np.isnan(result[key])]
            aggregated_results[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return aggregated_results
