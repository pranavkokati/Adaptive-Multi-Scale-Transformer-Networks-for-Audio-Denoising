#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import time
from pathlib import Path

sys.path.append('.')

from src.models import AdaptiveMultiScaleTransformer
from src.data.librispeech_dataset import LibriSpeechRealDataset
from src.evaluation import EvaluationSuite
from omegaconf import OmegaConf

def main():
    print("🚀 FAST TRAINING - NO WANDB BLOCKING")
    print("="*50)
    
    # Load config
    config = OmegaConf.load('configs/config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    
    # Create dataset
    data_root = Path(config.paths.data_root)
    dataset = LibriSpeechRealDataset(data_root, config.data.sample_rate, config.data.segment_length)
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = AdaptiveMultiScaleTransformer(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print("\n🔄 Starting Training...")
    model.train()
    
    for epoch in range(5):  # Quick training
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            if batch_idx >= 10:  # Limit batches for speed
                break
                
            noisy, clean = noisy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            enhanced = model(noisy)
            loss = criterion(enhanced, clean)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    # Save model
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / 'fast_trained_model.pt')
    
    print("\n✅ TRAINING COMPLETED!")
    print(f"Model saved to: {checkpoint_dir / 'fast_trained_model.pt'}")
    print("🎉 Ready for inference!")

if __name__ == "__main__":
    main()
