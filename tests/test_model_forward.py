import torch
from src.adaptive_transformer import AdaptiveMultiScaleTransformer

class DummyConfig:
    class model:
        hidden_dim = 32
        num_heads = 2
        base_layers = 2
        max_layers = 4
        num_scales = 2
        dropout = 0.1
        temperature = 0.07
    class data:
        n_fft = 64
        hop_length = 16
        win_length = 64
        sample_rate = 16000
        segment_length = 1.0
    class training:
        batch_size = 2
    class paths:
        checkpoint_dir = "checkpoints"
    class wandb:
        project = None
        entity = None
        tags = []


def test_model_forward():
    config = DummyConfig()
    model = AdaptiveMultiScaleTransformer(config)
    batch_size = 2
    seq_length = int(config.data.sample_rate * config.data.segment_length)
    noisy_audio = torch.randn(batch_size, seq_length)
    output = model(noisy_audio, mode='inference')
    assert 'enhanced_audio' in output
    assert output['enhanced_audio'].shape[0] == batch_size 