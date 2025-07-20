import torch
import torch.nn as nn

class LightweightEnhancer(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, input_dim, 3, padding=1)
        )
    def forward(self, audio):
        return self.net(audio.unsqueeze(1)).squeeze(1) 