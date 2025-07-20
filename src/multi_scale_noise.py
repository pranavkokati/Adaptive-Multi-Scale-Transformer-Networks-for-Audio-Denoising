import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        self.norm = nn.GroupNorm(8, out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class NoiseComplexityEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 4)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class MultiScaleNoiseCharacterization(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=512, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        
        self.scale_convs = nn.ModuleList()
        for i in range(num_scales):
            kernel_size = 2**(i+1) + 1
            dilation = 2**i
            self.scale_convs.append(
                DilatedConvBlock(input_channels, hidden_dim, kernel_size, dilation)
            )
        
        self.feature_fusion = nn.Conv1d(
            hidden_dim * num_scales, hidden_dim, 1
        )
        
        self.complexity_estimator = NoiseComplexityEstimator(
            hidden_dim, hidden_dim // 2
        )
        
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
    def estimate_distribution_parameters(self, features):
        mean = torch.mean(features, dim=-1, keepdim=True)
        variance = torch.var(features, dim=-1, keepdim=True)
        
        centered = features - mean
        skewness = torch.mean(centered**3, dim=-1, keepdim=True) / (variance**1.5 + 1e-8)
        kurtosis = torch.mean(centered**4, dim=-1, keepdim=True) / (variance**2 + 1e-8) - 3
        
        return torch.cat([mean, variance, skewness, kurtosis], dim=1)
    
    def forward(self, x):
        batch_size, channels, length = x.shape
        
        scale_features = []
        for i, conv in enumerate(self.scale_convs):
            features = conv(x)
            scale_features.append(features)
        
        combined_features = torch.cat(scale_features, dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        noise_params = self.estimate_distribution_parameters(fused_features)
        
        complexity_weights = F.softmax(self.scale_weights, dim=0)
        weighted_features = sum(
            w * feat for w, feat in zip(complexity_weights, scale_features)
        )
        
        pooled_features = F.adaptive_avg_pool1d(weighted_features, 1).squeeze(-1)
        complexity_scores = self.complexity_estimator(pooled_features)
        
        return {
            'multi_scale_features': scale_features,
            'fused_features': fused_features,
            'noise_parameters': noise_params,
            'complexity_scores': complexity_scores,
            'scale_weights': complexity_weights
        }
