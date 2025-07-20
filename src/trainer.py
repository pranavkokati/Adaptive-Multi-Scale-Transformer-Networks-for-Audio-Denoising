import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time

from src.adaptive_transformer import AdaptiveMultiScaleTransformer
from ..data import create_dataloader
from ..evaluation import EvaluationSuite


class Trainer:
    def __init__(self, config, model, device='cuda'):
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.training.learning_rate,
            total_steps=config.training.num_epochs * 1000,
            pct_start=0.1
        )
        
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.train_loader = create_dataloader(config, 'train', shuffle=True)
        self.val_loader = create_dataloader(config, 'val', shuffle=False)
        
        self.evaluator = EvaluationSuite(config.data.sample_rate)
        
        self.checkpoint_dir = Path(config.paths.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_pesq = 0.0
        self.current_epoch = 0
        
        if config.wandb.project:
            wandb.init(
                project=config.wandb.project,
                entity=config.wandb.entity,
                config=dict(config),
                tags=config.wandb.tags
            )
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_pesq': self.best_pesq,
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_pesq = checkpoint['best_pesq']
        self.current_epoch = checkpoint['epoch']
        
        return checkpoint['epoch']
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            clean_audio = batch['clean'].to(self.device)
            noisy_audio = batch['noisy'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                output = self.model(noisy_audio, clean_audio, mode='training')
                loss_dict = self.model.compute_loss(output, clean_audio, noisy_audio)
                loss = loss_dict['total_loss']
            
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}'
                })
                
                if wandb.run:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/reconstruction_loss': loss_dict['reconstruction_loss'].item(),
                        'train/magnitude_loss': loss_dict['magnitude_loss'].item(),
                        'train/contrastive_loss': loss_dict['contrastive_loss'].item(),
                        'train/learning_rate': current_lr,
                        'epoch': epoch,
                        'step': epoch * len(self.train_loader) + batch_idx
                    })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                clean_audio = batch['clean'].to(self.device)
                noisy_audio = batch['noisy'].to(self.device)
                
                with torch.cuda.amp.autocast():
                    output = self.model(noisy_audio, mode='inference')
                    loss_dict = self.model.compute_loss(output, clean_audio, noisy_audio)
                    loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
                
                metrics = self.evaluator.evaluate_batch(self.model, batch, self.device)
                all_metrics.append(metrics)
        
        avg_loss = total_loss / num_batches
        
        aggregated_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if not np.isnan(m[key])]
            aggregated_metrics[key] = np.mean(values)
        
        if wandb.run:
            log_dict = {'val/loss': avg_loss, 'epoch': epoch}
            for key, value in aggregated_metrics.items():
                log_dict[f'val/{key}'] = value
            wandb.log(log_dict)
        
        return avg_loss, aggregated_metrics
    
    def train(self):
        print(f"Starting training for {self.config.training.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if hasattr(self.model, 'setup_contrastive_learning'):
            self.model.setup_contrastive_learning()
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val PESQ: {val_metrics.get('pesq', 0):.3f}, Val STOI: {val_metrics.get('stoi', 0):.3f}")
            print(f"Val SI-SDR: {val_metrics.get('si_sdr', 0):.2f} dB, Epoch Time: {epoch_time:.1f}s")
            
            is_best = val_metrics.get('pesq', 0) > self.best_pesq
            if is_best:
                self.best_pesq = val_metrics.get('pesq', 0)
                print(f"New best PESQ: {self.best_pesq:.3f}")
            
            if epoch % self.config.training.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            self.current_epoch = epoch + 1
        
        print("Training completed!")
        return self.best_pesq


class InferenceEngine:
    def __init__(self, model_path, config, device='cuda'):
        self.device = device
        self.config = config
        
        self.model = AdaptiveMultiScaleTransformer(config).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.evaluator = EvaluationSuite(config.data.sample_rate)
    
    def enhance_audio(self, noisy_audio):
        self.model.eval()
        
        if isinstance(noisy_audio, np.ndarray):
            noisy_audio = torch.from_numpy(noisy_audio).float()
        
        if noisy_audio.dim() == 1:
            noisy_audio = noisy_audio.unsqueeze(0)
        
        noisy_audio = noisy_audio.to(self.device)
        
        with torch.no_grad():
            output = self.model(noisy_audio, mode='inference')
            enhanced_audio = output['enhanced_audio']
        
        return enhanced_audio.cpu().numpy()
    
    def enhance_file(self, input_path, output_path):
        import torchaudio
        import soundfile as sf
        
        audio, sr = torchaudio.load(input_path)
        
        if sr != self.config.data.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.data.sample_rate)
            audio = resampler(audio)
        
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        enhanced = self.enhance_audio(audio.squeeze(0))
        
        sf.write(output_path, enhanced.squeeze(), self.config.data.sample_rate)
        
        return output_path
    
    def batch_enhance(self, input_dir, output_dir):
        import glob
        from pathlib import Path
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_files = glob.glob(str(input_dir / "*.wav"))
        
        results = []
        for file_path in tqdm(audio_files, desc="Enhancing audio files"):
            file_name = Path(file_path).name
            output_path = output_dir / file_name
            
            enhanced_path = self.enhance_file(file_path, output_path)
            results.append(enhanced_path)
        
        return results
