#!/usr/bin/env python3

import torch
import hydra
from omegaconf import DictConfig
import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.training import InferenceEngine
from src.adaptive_transformer import AdaptiveMultiScaleTransformer
from src.metrics import EvaluationSuite
from src.data.real_datasets import create_real_dataloader
from src import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate audio denoising model with real data')
    parser.add_argument('--enhancement_methods', nargs='+', 
                       default=['transformer', 'diffusion', 'hybrid', 'multitask', 'lightweight'],
                       help='Enhancement methods to evaluate')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

def evaluate_methods(config, enhancement_methods):
    """Evaluate multiple enhancement methods and compare results."""
    results = {}
    
    for method in enhancement_methods:
        print(f"\nEvaluating {method} enhancement method...")
        config['enhancement_method'] = method
        
        # Initialize model with current method
        model = AdaptiveMultiScaleTransformer(config)
        
        # Evaluate
        method_results = evaluate_model(model, config)
        results[method] = method_results
        
        print(f"{method} Results:")
        for metric, value in method_results.items():
            print(f"  {metric}: {value:.3f}")
    
    return results

def main():
    args = parse_args()
    set_global_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Evaluating multiple enhancement methods...")
    print("REAL DATA ONLY - NO SYNTHETIC DATA USED")
    
    results = evaluate_methods(config, args.enhancement_methods)
    
    # Print comparison table
    print("\n" + "="*60)
    print("ENHANCEMENT METHOD COMPARISON")
    print("="*60)
    metrics = ['pesq', 'stoi', 'si_sdr', 'snr']
    print(f"{'Method':<15}", end="")
    for metric in metrics:
        print(f"{metric.upper():<10}", end="")
    print()
    print("-" * 60)
    
    for method, method_results in results.items():
        print(f"{method:<15}", end="")
        for metric in metrics:
            value = method_results.get(metric, 0.0)
            print(f"{value:<10.3f}", end="")
        print()
    
    print("="*60)


if __name__ == "__main__":
    main()
