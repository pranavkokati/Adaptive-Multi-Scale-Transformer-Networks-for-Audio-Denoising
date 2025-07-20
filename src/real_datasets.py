import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchaudio
import numpy as np
import os
import glob
from pathlib import Path
import soundfile as sf
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm
import json


class RealDatasetManager:
    def __init__(self, data_root, sample_rate=16000, segment_length=4.0):
        self.data_root = Path(data_root)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.datasets = []
        
    def download_and_prepare_real_datasets(self):
        print("Downloading and preparing real datasets only...")
        
        success_count = 0
        
        if self.download_voicebank_demand():
            success_count += 1
            
        if self.download_microsoft_dns():
            success_count += 1
            
        if success_count == 0:
            raise RuntimeError("Failed to download any real datasets. Please check your internet connection.")
        
        return self.create_combined_real_dataset()
    
    def download_voicebank_demand(self):
        print("Downloading VoiceBank+DEMAND dataset...")
        
        dataset_dir = self.data_root / "voicebank_demand"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        base_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791"
        
        files_to_download = [
            ("clean_trainset_wav.zip", "clean_trainset_wav.zip"),
            ("noisy_trainset_wav.zip", "noisy_trainset_wav.zip"),
            ("clean_testset_wav.zip", "clean_testset_wav.zip"),
            ("noisy_testset_wav.zip", "noisy_testset_wav.zip")
        ]
        
        success = True
        for filename, local_filename in files_to_download:
            local_path = dataset_dir / local_filename
            extract_dir = dataset_dir / filename.replace('.zip', '')
            
            if not extract_dir.exists():
                if not local_path.exists():
                    print(f"Downloading {filename}...")
                    try:
                        url = f"{base_url}/{filename}?sequence=1&isAllowed=y"
                        self._download_with_progress(url, local_path)
                    except Exception as e:
                        print(f"Failed to download {filename}: {e}")
                        success = False
                        continue
                
                print(f"Extracting {filename}...")
                try:
                    with zipfile.ZipFile(local_path, 'r') as zip_ref:
                        zip_ref.extractall(dataset_dir)
                    local_path.unlink()
                except Exception as e:
                    print(f"Failed to extract {filename}: {e}")
                    success = False
        
        if success:
            print("VoiceBank+DEMAND dataset setup completed!")
        return success
    
    def download_microsoft_dns(self):
        print("Attempting to download Microsoft DNS Challenge dataset...")
        
        dataset_dir = self.data_root / "microsoft_dns"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        dns_urls = [
            "https://github.com/microsoft/DNS-Challenge/releases/download/v4.0/datasets_fullband.tar.bz2",
            "https://github.com/microsoft/DNS-Challenge/releases/download/v4.0/clean_fullband.tar.bz2",
            "https://github.com/microsoft/DNS-Challenge/releases/download/v4.0/noise_fullband.tar.bz2"
        ]
        
        success = False
        for url in dns_urls:
            filename = url.split('/')[-1]
            local_path = dataset_dir / filename
            
            if not local_path.exists():
                try:
                    print(f"Downloading {filename}...")
                    self._download_with_progress(url, local_path)
                    
                    print(f"Extracting {filename}...")
                    with tarfile.open(local_path, 'r:bz2') as tar:
                        tar.extractall(dataset_dir)
                    
                    local_path.unlink()
                    success = True
                    break
                except Exception as e:
                    print(f"Failed to download {filename}: {e}")
                    continue
        
        if not success:
            print("Microsoft DNS dataset download failed - this is expected as it requires registration")
            print("You can manually download from: https://github.com/microsoft/DNS-Challenge")
        
        return success
    
    def _download_with_progress(self, url, output_path):
        class ProgressBar:
            def __init__(self):
                self.pbar = None
            
            def __call__(self, block_num, block_size, total_size):
                if not self.pbar:
                    self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
                downloaded = block_num * block_size
                if downloaded < total_size:
                    self.pbar.update(block_size)
                else:
                    self.pbar.close()
        
        urllib.request.urlretrieve(url, output_path, ProgressBar())
    
    def create_combined_real_dataset(self):
        datasets = []
        
        voicebank_dir = self.data_root / "voicebank_demand"
        if voicebank_dir.exists():
            try:
                from .dataset import AudioDataset
                train_dataset = AudioDataset(
                    str(voicebank_dir), 'train', self.sample_rate, self.segment_length
                )
                if len(train_dataset) > 0:
                    datasets.append(train_dataset)
                    print(f"Added VoiceBank+DEMAND: {len(train_dataset)} samples")
            except Exception as e:
                print(f"Failed to load VoiceBank+DEMAND: {e}")
        
        dns_dir = self.data_root / "microsoft_dns"
        if dns_dir.exists() and len(list(dns_dir.glob("**/*.wav"))) > 0:
            try:
                dns_dataset = MicrosoftDNSDataset(
                    str(dns_dir), self.sample_rate, self.segment_length
                )
                if len(dns_dataset) > 0:
                    datasets.append(dns_dataset)
                    print(f"Added Microsoft DNS: {len(dns_dataset)} samples")
            except Exception as e:
                print(f"Failed to load Microsoft DNS: {e}")
        
        if datasets:
            combined_dataset = ConcatDataset(datasets)
            print(f"Total combined real dataset: {len(combined_dataset)} samples")
            return combined_dataset
        else:
            raise RuntimeError("No real datasets loaded successfully. Please check dataset availability.")


class MicrosoftDNSDataset(Dataset):
    def __init__(self, data_root, sample_rate=16000, segment_length=4.0):
        self.data_root = Path(data_root)
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        
        self.clean_files = []
        self.noisy_files = []
        
        self._find_dns_files()
        
        if len(self.clean_files) == 0:
            raise ValueError("No Microsoft DNS files found")
    
    def _find_dns_files(self):
        clean_patterns = ["**/clean/**/*.wav", "**/clean_*.wav"]
        noisy_patterns = ["**/noisy/**/*.wav", "**/noisy_*.wav", "**/mix/**/*.wav"]
        
        for pattern in clean_patterns:
            self.clean_files.extend(glob.glob(str(self.data_root / pattern), recursive=True))
        
        for pattern in noisy_patterns:
            self.noisy_files.extend(glob.glob(str(self.data_root / pattern), recursive=True))
        
        self.clean_files = sorted(self.clean_files)
        self.noisy_files = sorted(self.noisy_files)
        
        min_length = min(len(self.clean_files), len(self.noisy_files))
        self.clean_files = self.clean_files[:min_length]
        self.noisy_files = self.noisy_files[:min_length]
    
    def _load_audio(self, file_path):
        try:
            audio, sr = torchaudio.load(file_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            return audio.squeeze(0)
        except Exception:
            audio, sr = sf.read(file_path)
            audio = torch.from_numpy(audio).float()
            
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio.unsqueeze(0)).squeeze(0)
            
            if len(audio.shape) > 1:
                audio = torch.mean(audio, dim=-1)
            
            return audio
    
    def _segment_audio(self, audio):
        if len(audio) >= self.segment_samples:
            start_idx = torch.randint(0, len(audio) - self.segment_samples + 1, (1,)).item()
            return audio[start_idx:start_idx + self.segment_samples]
        else:
            padding = self.segment_samples - len(audio)
            return F.pad(audio, (0, padding), mode='constant', value=0)
    
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        clean_audio = self._load_audio(self.clean_files[idx])
        noisy_audio = self._load_audio(self.noisy_files[idx])
        
        clean_segment = self._segment_audio(clean_audio)
        noisy_segment = self._segment_audio(noisy_audio)
        
        clean_segment = clean_segment / (torch.max(torch.abs(clean_segment)) + 1e-8)
        noisy_segment = noisy_segment / (torch.max(torch.abs(noisy_segment)) + 1e-8)
        
        return {
            'clean': clean_segment,
            'noisy': noisy_segment,
            'file_id': Path(self.clean_files[idx]).stem
        }


def create_real_dataloader(config, subset='train', shuffle=True):
    data_root = Path(config.paths.data_root)
    manager = RealDatasetManager(data_root, config.data.sample_rate, config.data.segment_length)
    
    combined_dataset = manager.download_and_prepare_real_datasets()
    
    total_size = len(combined_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    from torch.utils.data import random_split
    train_dataset, val_dataset, test_dataset = random_split(
        combined_dataset, [train_size, val_size, test_size]
    )
    
    if subset == 'train':
        dataset = train_dataset
    elif subset == 'val':
        dataset = val_dataset
    else:
        dataset = test_dataset
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True if subset == 'train' else False
    )
    
    return dataloader
