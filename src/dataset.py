import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import os
import glob
from pathlib import Path
import soundfile as sf


class AudioDataset(Dataset):
    def __init__(self, data_root, subset='train', sample_rate=16000, segment_length=4.0):
        self.data_root = Path(data_root)
        self.subset = subset
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        
        self.clean_files = []
        self.noisy_files = []
        
        self._load_file_paths()
        
    def _load_file_paths(self):
        if 'voicebank' in str(self.data_root).lower():
            self._load_voicebank_paths()
        elif 'wham' in str(self.data_root).lower():
            self._load_wham_paths()
        else:
            raise ValueError(f"Unsupported dataset format in {self.data_root}")
    
    def _load_voicebank_paths(self):
        clean_dir = self.data_root / 'clean_trainset_wav' if self.subset == 'train' else self.data_root / 'clean_testset_wav'
        noisy_dir = self.data_root / 'noisy_trainset_wav' if self.subset == 'train' else self.data_root / 'noisy_testset_wav'
        
        if clean_dir.exists() and noisy_dir.exists():
            clean_files = sorted(glob.glob(str(clean_dir / '*.wav')))
            noisy_files = sorted(glob.glob(str(noisy_dir / '*.wav')))
            
            self.clean_files = clean_files
            self.noisy_files = noisy_files
    
    def _load_wham_paths(self):
        subset_dir = self.data_root / self.subset
        
        if subset_dir.exists():
            clean_files = sorted(glob.glob(str(subset_dir / 'clean' / '*.wav')))
            noisy_files = sorted(glob.glob(str(subset_dir / 'mix' / '*.wav')))
            
            self.clean_files = clean_files
            self.noisy_files = noisy_files
    
    def _load_audio(self, file_path):
        try:
            audio, sr = torchaudio.load(file_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            return audio.squeeze(0)
        except Exception as e:
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
        
        min_length = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_length]
        noisy_audio = noisy_audio[:min_length]
        
        clean_segment = self._segment_audio(clean_audio)
        noisy_segment = self._segment_audio(noisy_audio)
        
        clean_segment = clean_segment / (torch.max(torch.abs(clean_segment)) + 1e-8)
        noisy_segment = noisy_segment / (torch.max(torch.abs(noisy_segment)) + 1e-8)
        
        return {
            'clean': clean_segment,
            'noisy': noisy_segment,
            'file_id': Path(self.clean_files[idx]).stem
        }


class DataAugmentation:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def add_noise(self, audio, snr_db_range=(5, 20)):
        snr_db = torch.uniform(snr_db_range[0], snr_db_range[1], (1,)).item()
        signal_power = torch.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(audio) * torch.sqrt(noise_power)
        return audio + noise
    
    def time_shift(self, audio, max_shift_ratio=0.1):
        max_shift = int(len(audio) * max_shift_ratio)
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        
        if shift > 0:
            return F.pad(audio[:-shift], (shift, 0), mode='constant', value=0)
        elif shift < 0:
            return F.pad(audio[-shift:], (0, -shift), mode='constant', value=0)
        else:
            return audio
    
    def amplitude_scaling(self, audio, scale_range=(0.8, 1.2)):
        scale = torch.uniform(scale_range[0], scale_range[1], (1,)).item()
        return audio * scale
    
    def apply_augmentation(self, audio, augment_prob=0.5):
        if torch.rand(1) < augment_prob:
            if torch.rand(1) < 0.3:
                audio = self.add_noise(audio)
            if torch.rand(1) < 0.3:
                audio = self.time_shift(audio)
            if torch.rand(1) < 0.3:
                audio = self.amplitude_scaling(audio)
        
        return audio


def create_dataloader(config, subset='train', shuffle=True):
    dataset = AudioDataset(
        data_root=config.paths.data_root,
        subset=subset,
        sample_rate=config.data.sample_rate,
        segment_length=config.data.segment_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True if subset == 'train' else False
    )
    
    return dataloader


def download_voicebank_demand():
    import urllib.request
    import zipfile
    import shutil
    
    base_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791"
    
    files_to_download = [
        ("clean_trainset_wav.zip", "clean_trainset_wav.zip"),
        ("noisy_trainset_wav.zip", "noisy_trainset_wav.zip"),
        ("clean_testset_wav.zip", "clean_testset_wav.zip"),
        ("noisy_testset_wav.zip", "noisy_testset_wav.zip")
    ]
    
    data_dir = Path("./data/voicebank_demand")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, local_filename in files_to_download:
        url = f"{base_url}/{filename}"
        local_path = data_dir / local_filename
        
        if not local_path.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, local_path)
            
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            local_path.unlink()
    
    print("VoiceBank+DEMAND dataset downloaded and extracted successfully!")


def setup_data_directories():
    directories = [
        "./data",
        "./data/voicebank_demand",
        "./data/wham",
        "./checkpoints",
        "./logs",
        "./outputs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Data directories created successfully!")
