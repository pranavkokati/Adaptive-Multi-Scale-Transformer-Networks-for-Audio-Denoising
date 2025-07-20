import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        output = self.out_proj(context)
        return output, attn_weights


class CrossModalAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.time_to_freq_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.freq_to_time_attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        self.time_norm1 = nn.LayerNorm(hidden_dim)
        self.time_norm2 = nn.LayerNorm(hidden_dim)
        self.freq_norm1 = nn.LayerNorm(hidden_dim)
        self.freq_norm2 = nn.LayerNorm(hidden_dim)
        
        self.time_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.freq_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        
    def forward(self, time_features, freq_features):
        time_attended, time_attn_weights = self.time_to_freq_attn(
            time_features, freq_features, freq_features
        )
        time_features = self.time_norm1(time_features + time_attended)
        time_features = self.time_norm2(time_features + self.time_ffn(time_features))
        
        freq_attended, freq_attn_weights = self.freq_to_time_attn(
            freq_features, time_features, time_features
        )
        freq_features = self.freq_norm1(freq_features + freq_attended)
        freq_features = self.freq_norm2(freq_features + self.freq_ffn(freq_features))
        
        fusion_weights = F.softmax(self.fusion_weights, dim=0)
        fused_features = fusion_weights[0] * time_features + fusion_weights[1] * freq_features
        
        return {
            'time_features': time_features,
            'freq_features': freq_features,
            'fused_features': fused_features,
            'time_attention': time_attn_weights,
            'freq_attention': freq_attn_weights
        }


class ProgressiveCrossModalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers=6, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.time_embedding = nn.Linear(hidden_dim, hidden_dim)
        self.freq_embedding = nn.Linear(hidden_dim, hidden_dim)
        
        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.progressive_weights = nn.Parameter(torch.ones(num_layers))
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, time_features, freq_features):
        time_features = self.time_embedding(time_features)
        freq_features = self.freq_embedding(freq_features)
        
        progressive_outputs = []
        attention_maps = []
        
        for i, layer in enumerate(self.cross_attention_layers):
            output = layer(time_features, freq_features)
            time_features = output['time_features']
            freq_features = output['freq_features']
            
            progressive_outputs.append(output['fused_features'])
            attention_maps.append({
                'time_attention': output['time_attention'],
                'freq_attention': output['freq_attention']
            })
        
        progressive_weights = F.softmax(self.progressive_weights, dim=0)
        weighted_output = sum(
            w * output for w, output in zip(progressive_weights, progressive_outputs)
        )
        
        final_output = self.output_projection(weighted_output)
        
        return {
            'output': final_output,
            'progressive_outputs': progressive_outputs,
            'attention_maps': attention_maps,
            'progressive_weights': progressive_weights
        }
