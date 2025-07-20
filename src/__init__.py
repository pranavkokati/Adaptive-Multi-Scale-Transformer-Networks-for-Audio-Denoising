from .adaptive_transformer import AdaptiveMultiScaleTransformer
from .multi_scale_noise import MultiScaleNoiseCharacterization
from .cross_attention import ProgressiveCrossModalAttention
from .contrastive_learning import SelfSupervisedContrastiveLearning
from .adaptive_scaling import AdaptiveComputationAllocation

import random
import numpy as np
import torch

def set_global_seed(seed: int = 42):
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

__all__ = [
    'AdaptiveMultiScaleTransformer',
    'MultiScaleNoiseCharacterization',
    'ProgressiveCrossModalAttention',
    'SelfSupervisedContrastiveLearning',
    'AdaptiveComputationAllocation'
]
