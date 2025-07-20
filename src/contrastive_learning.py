import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MomentumEncoder(nn.Module):
    def __init__(self, encoder, momentum=0.999):
        super().__init__()
        self.encoder = encoder
        self.momentum_encoder = self._create_momentum_encoder(encoder)
        self.momentum = momentum
        
    def _create_momentum_encoder(self, encoder):
        momentum_encoder = type(encoder)(
            **{k: v for k, v in encoder.__dict__.items() 
               if not k.startswith('_') and k != 'training'}
        )
        for param in momentum_encoder.parameters():
            param.requires_grad = False
        return momentum_encoder
    
    @torch.no_grad()
    def _update_momentum_encoder(self):
        for param_q, param_k in zip(self.encoder.parameters(), 
                                   self.momentum_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
    
    def forward(self, x, use_momentum=False):
        if use_momentum:
            with torch.no_grad():
                self._update_momentum_encoder()
                return self.momentum_encoder(x)
        else:
            return self.encoder(x)


class TemporalAugmentation(nn.Module):
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        
    def time_shift(self, audio, max_shift=0.1):
        batch_size, length = audio.shape
        shift_samples = int(max_shift * self.sample_rate)
        shifts = torch.randint(-shift_samples, shift_samples + 1, (batch_size,))
        
        shifted_audio = torch.zeros_like(audio)
        for i, shift in enumerate(shifts):
            if shift > 0:
                shifted_audio[i, shift:] = audio[i, :-shift]
            elif shift < 0:
                shifted_audio[i, :shift] = audio[i, -shift:]
            else:
                shifted_audio[i] = audio[i]
        
        return shifted_audio
    
    def time_stretch(self, audio, stretch_factor_range=(0.8, 1.2)):
        batch_size, length = audio.shape
        stretch_factors = torch.uniform(
            stretch_factor_range[0], stretch_factor_range[1], (batch_size,)
        )
        
        stretched_audio = torch.zeros_like(audio)
        for i, factor in enumerate(stretch_factors):
            new_length = int(length / factor)
            if new_length <= length:
                indices = torch.linspace(0, length - 1, new_length).long()
                stretched_audio[i, :new_length] = audio[i, indices]
            else:
                indices = torch.linspace(0, length - 1, length).long()
                stretched_audio[i] = audio[i, indices]
        
        return stretched_audio
    
    def add_noise(self, audio, noise_level=0.01):
        noise = torch.randn_like(audio) * noise_level
        return audio + noise
    
    def forward(self, audio):
        augmented = audio.clone()
        
        if torch.rand(1) < 0.5:
            augmented = self.time_shift(augmented)
        
        if torch.rand(1) < 0.3:
            augmented = self.time_stretch(augmented)
        
        if torch.rand(1) < 0.4:
            augmented = self.add_noise(augmented)
        
        return augmented


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1)


class SelfSupervisedContrastiveLearning(nn.Module):
    def __init__(self, encoder, hidden_dim=512, projection_dim=128, 
                 temperature=0.07, queue_size=65536, momentum=0.999):
        super().__init__()
        self.encoder = encoder
        self.momentum_encoder = MomentumEncoder(encoder, momentum)
        self.projection_head = ProjectionHead(hidden_dim, hidden_dim // 2, projection_dim)
        self.momentum_projection = ProjectionHead(hidden_dim, hidden_dim // 2, projection_dim)
        
        self.temperature = temperature
        self.queue_size = queue_size
        
        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.augmentation = TemporalAugmentation()
        
        for param in self.momentum_projection.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def info_nce_loss(self, query, positive_key, negative_keys):
        positive_logits = torch.sum(query * positive_key, dim=-1, keepdim=True) / self.temperature
        negative_logits = torch.mm(query, negative_keys) / self.temperature
        
        logits = torch.cat([positive_logits, negative_logits], dim=1)
        labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        
        return F.cross_entropy(logits, labels)
    
    def forward(self, clean_audio, noisy_audio):
        batch_size = clean_audio.shape[0]
        
        augmented_clean = self.augmentation(clean_audio)
        augmented_noisy = self.augmentation(noisy_audio)
        
        clean_features = self.encoder(clean_audio)
        noisy_features = self.encoder(noisy_audio)
        
        clean_proj = self.projection_head(clean_features.mean(dim=-1))
        noisy_proj = self.projection_head(noisy_features.mean(dim=-1))
        
        with torch.no_grad():
            augmented_clean_features = self.momentum_encoder(augmented_clean, use_momentum=True)
            augmented_noisy_features = self.momentum_encoder(augmented_noisy, use_momentum=True)
            
            augmented_clean_proj = self.momentum_projection(augmented_clean_features.mean(dim=-1))
            augmented_noisy_proj = self.momentum_projection(augmented_noisy_features.mean(dim=-1))
            
            self._dequeue_and_enqueue(augmented_clean_proj)
        
        clean_loss = self.info_nce_loss(clean_proj, augmented_clean_proj, self.queue.T)
        noisy_loss = self.info_nce_loss(noisy_proj, augmented_noisy_proj, self.queue.T)
        
        contrastive_loss = (clean_loss + noisy_loss) / 2
        
        return {
            'contrastive_loss': contrastive_loss,
            'clean_features': clean_features,
            'noisy_features': noisy_features,
            'clean_projections': clean_proj,
            'noisy_projections': noisy_proj
        }
