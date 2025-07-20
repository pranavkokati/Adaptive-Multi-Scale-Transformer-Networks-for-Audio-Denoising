import importlib
import pytest

MODULES = [
    'src.adaptive_scaling',
    'src.adaptive_transformer',
    'src.contrastive_learning',
    'src.cross_attention',
    'src.dataset',
    'src.librispeech_dataset',
    'src.metrics',
    'src.multi_scale_noise',
    'src.real_datasets',
    'src.research_visualizations',
    'src.trainer',
    'src.diffusion_enhancer',
    'src.hybrid_denoiser',
    'src.multitask_pipeline',
    'src.lightweight_model',
    'src.pretrained_audio',
]

def test_imports():
    for module in MODULES:
        importlib.import_module(module) 