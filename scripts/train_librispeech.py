#!/usr/bin/env python3

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.models import AdaptiveMultiScaleTransformer
from src.training import Trainer
from src.data import setup_data_directories
from src.data.librispeech_dataset import create_librispeech_dataloader
from src.evaluation import EvaluationSuite
from src import set_global_seed


class LibriSpeechTrainer(Trainer):
    def __init__(self, config, model, device='cuda'):
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        print("Setting up LibriSpeech real dataset...")
        self.train_loader = create_librispeech_dataloader(config, 'train', shuffle=True)
        self.val_loader = create_librispeech_dataloader(config, 'val', shuffle=False)
        
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
                tags=config.wandb.tags + ["librispeech", "real_data_only"]
            )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    print("Adaptive Multi-Scale Transformer Networks - LibriSpeech Real Data Training")
    print("=" * 70)
    print("USING LIBRISPEECH REAL SPEECH DATA - NO SYNTHETIC DATA")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    setup_data_directories()
    
    print("\n" + "="*50)
    print("PHASE 1: LibriSpeech Real Dataset Setup")
    print("="*50)
    
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
    
    trainer = LibriSpeechTrainer(config, model, device)
    
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
    print("PHASE 4: Training Execution (LibriSpeech Real Data)")
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
        print("PHASE 5: Final Evaluation on LibriSpeech")
        print("="*50)
        
        test_loader = create_librispeech_dataloader(config, 'test', shuffle=False)
        
        final_results = trainer.evaluator.evaluate_dataset(
            trainer.model, test_loader, device, max_batches=20
        )
        
        print("Final Test Results (LibriSpeech Real Data):")
        for metric, stats in final_results.items():
            if metric in ['pesq', 'stoi', 'si_sdr', 'snr']:
                print(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nLibriSpeech real data training pipeline completed successfully!")
    print("You can now use the trained model for inference:")
    print(f"  python scripts/inference_real.py --checkpoint {trainer.checkpoint_dir / 'best_model.pt'} --input your_audio.wav --output enhanced.wav")


if __name__ == "__main__":
    set_global_seed(42)
    main()
