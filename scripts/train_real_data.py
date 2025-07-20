#!/usr/bin/env python3

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
from pathlib import Path
import time
import argparse
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from src.adaptive_transformer import AdaptiveMultiScaleTransformer
from src.training import Trainer
from src.data import setup_data_directories
from src.data.real_datasets import create_real_dataloader, RealDatasetManager
from src.evaluation import EvaluationSuite
from src import set_global_seed


class RealDataTrainer(Trainer):
    def __init__(self, config, model, device='cuda'):
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        print("Setting up real datasets only...")
        self.train_loader = create_real_dataloader(config, 'train', shuffle=True)
        self.val_loader = create_real_dataloader(config, 'val', shuffle=False)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.training.learning_rate,
            total_steps=config.training.num_epochs * len(self.train_loader),
            pct_start=0.1
        )
        
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.evaluator = EvaluationSuite(config.data.sample_rate)
        
        self.checkpoint_dir = Path(config.paths.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_pesq = 0.0
        self.current_epoch = 0
        
        if config.wandb.project:
            import wandb
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                config=dict(config),
                tags=config.wandb.tags + ["real_data_only", "no_synthetic"]
            )


def parse_args():
    parser = argparse.ArgumentParser(description='Train audio denoising model with real data')
    parser.add_argument('--enhancement_method', type=str, default='transformer',
                       choices=['transformer', 'diffusion', 'hybrid', 'multitask', 'lightweight'],
                       help='Enhancement method to use')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    args = parse_args()
    set_global_seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line args
    config['enhancement_method'] = args.enhancement_method
    
    print(f"Training with enhancement method: {args.enhancement_method}")
    print("REAL DATA ONLY - NO SYNTHETIC DATA USED")
    
    print("Adaptive Multi-Scale Transformer Networks - Real Data Only Training")
    print("=" * 70)
    print("NO SYNTHETIC DATA - REAL DATASETS ONLY")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    setup_data_directories()
    
    print("\n" + "="*50)
    print("PHASE 1: Real Dataset Preparation")
    print("="*50)
    
    data_root = Path(config.paths.data_root)
    manager = RealDatasetManager(data_root, config.data.sample_rate, config.data.segment_length)
    
    try:
        combined_dataset = manager.download_and_prepare_real_datasets()
        print(f"Successfully loaded {len(combined_dataset)} real audio samples")
    except Exception as e:
        print(f"Failed to load real datasets: {e}")
        print("Please ensure you have internet access and the datasets are available.")
        return
    
    print("\n" + "="*50)
    print("PHASE 2: Model Initialization")
    print("="*50)
    
    model = AdaptiveMultiScaleTransformer(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.1f} MB")
    
    print("\n" + "="*50)
    print("PHASE 3: Training Setup")
    print("="*50)
    
    trainer = RealDataTrainer(config, model, device)
    
    if hasattr(trainer.model, 'setup_contrastive_learning'):
        trainer.model.setup_contrastive_learning()
        print("Contrastive learning enabled")
    
    print(f"Training configuration:")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Training batches: {len(trainer.train_loader)}")
    print(f"  Validation batches: {len(trainer.val_loader)}")
    
    print("\n" + "="*50)
    print("PHASE 4: Training Execution (Real Data Only)")
    print("="*50)
    
    start_time = time.time()
    
    try:
        best_pesq = trainer.train()
        
        training_time = time.time() - start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Best PESQ achieved: {best_pesq:.3f}")
        print(f"Total training time: {hours}h {minutes}m")
        print(f"Model saved to: {trainer.checkpoint_dir / 'best_model.pt'}")
        
        print("\n" + "="*50)
        print("PHASE 5: Final Evaluation on Real Data")
        print("="*50)
        
        test_loader = create_real_dataloader(config, 'test', shuffle=False)
        
        final_results = trainer.evaluator.evaluate_dataset(
            trainer.model, test_loader, device, max_batches=50
        )
        
        print("Final Test Results (Real Data Only):")
        for metric, stats in final_results.items():
            if metric in ['pesq', 'stoi', 'si_sdr', 'snr']:
                print(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nReal data training pipeline completed successfully!")
    print("You can now use the trained model for inference:")
    print(f"  python scripts/inference_real.py --checkpoint {trainer.checkpoint_dir / 'best_model.pt'} --input your_audio.wav --output enhanced.wav")


if __name__ == "__main__":
    main()
