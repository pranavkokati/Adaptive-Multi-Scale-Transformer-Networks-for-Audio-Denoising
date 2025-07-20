# Data Sources and Acquisition

**This document details all real audio datasets used in this project. NO SYNTHETIC DATA IS GENERATED OR USED.**

## Overview

All datasets used in this project are **real, recorded audio** from verified sources. We do not generate synthetic noise or speech - every audio sample is an actual recording from the real world.

## Dataset Sources

### 1. VoiceBank+DEMAND Dataset

**Source**: Edinburgh DataShare (University of Edinburgh)
- **URL**: https://datashare.ed.ac.uk/handle/10283/2791
- **License**: Creative Commons Attribution 4.0
- **Paper**: https://arxiv.org/abs/1804.03619

**Contents**:
- **Clean Speech**: 28 speakers reading sentences from VoiceBank corpus
- **Noise**: Real environmental noise from DEMAND dataset
- **Format**: 16kHz WAV files
- **Duration**: 1-10 seconds per file
- **Total**: 11,572 files (5,764 train, 5,808 test)

**Verification**:
```bash
# Download and verify
python scripts/download_datasets.py --dataset voicebank_demand
python scripts/verify_data.py --dataset voicebank_demand
```

### 2. LibriSpeech Dataset

**Source**: OpenSLR (Open Speech and Language Resources)
- **URL**: https://www.openslr.org/12/
- **License**: Creative Commons Attribution 4.0
- **Paper**: https://arxiv.org/abs/1412.5567

**Contents**:
- **Clean Speech**: High-quality audiobook recordings
- **Speakers**: 2,484 speakers
- **Format**: 16kHz FLAC files
- **Duration**: 1-30 seconds per file
- **Total**: 1,000 hours of speech

**Verification**:
```bash
# Download and verify
python scripts/download_datasets.py --dataset librispeech
python scripts/verify_data.py --dataset librispeech
```

### 3. Microsoft DNS Challenge Dataset

**Source**: Microsoft DNS Challenge
- **URL**: https://github.com/microsoft/DNS-Challenge
- **License**: MIT License
- **Paper**: https://arxiv.org/abs/2001.08662

**Contents**:
- **Noise**: Real-world environmental noise recordings
- **Categories**: Office, cafe, street, home, etc.
- **Format**: 16kHz WAV files
- **Duration**: 1-60 seconds per file
- **Total**: ~65,000 noise files

**Verification**:
```bash
# Download and verify
python scripts/download_datasets.py --dataset microsoft_dns
python scripts/verify_data.py --dataset microsoft_dns
```

### 4. Freesound Public Domain Noise

**Source**: Freesound.org
- **URL**: https://freesound.org/
- **License**: Public Domain
- **API**: https://freesound.org/docs/api/

**Contents**:
- **Noise**: Diverse environmental sounds
- **Categories**: Traffic, nature, machinery, etc.
- **Format**: Various formats, converted to 16kHz WAV
- **Duration**: 1-30 seconds per file
- **Total**: ~1,000 noise files

**Verification**:
```bash
# Download and verify
python scripts/download_datasets.py --dataset freesound_noise
python scripts/verify_data.py --dataset freesound_noise
```

## Data Acquisition Process

### Automated Download

Our download script (`scripts/download_datasets.py`) handles:

1. **URL Verification**: Validates download URLs
2. **Checksum Verification**: SHA256 checksums for integrity
3. **Size Verification**: File size validation
4. **Audio Validation**: Loads and verifies audio files
5. **Synthetic Detection**: Checks for artificial patterns

### Manual Download (if needed)

If automated download fails, manual download instructions:

#### VoiceBank+DEMAND
```bash
# Visit: https://datashare.ed.ac.uk/handle/10283/2791
# Download: clean_trainset_wav.zip, noisy_trainset_wav.zip, etc.
# Extract to: data/voicebank_demand/
```

#### LibriSpeech
```bash
# Visit: https://www.openslr.org/12/
# Download: dev-clean.tar.gz
# Extract to: data/librispeech/
```

#### Microsoft DNS
```bash
# Visit: https://github.com/microsoft/DNS-Challenge
# Download: noise_fullband.zip
# Extract to: data/microsoft_dns/
```

## Data Verification

### Automated Verification

Our verification script (`scripts/verify_data.py`) performs:

1. **Structure Validation**: Checks directory structure
2. **File Count**: Verifies expected number of files
3. **Audio Quality**: Validates sample rate, duration, format
4. **Synthetic Detection**: Identifies artificial patterns
5. **Corruption Check**: Detects corrupted files

### Manual Verification

You can manually verify datasets:

```bash
# Check file structure
ls -la data/voicebank_demand/
ls -la data/librispeech/
ls -la data/microsoft_dns/

# Check audio files
python -c "
import torchaudio
import os
for root, dirs, files in os.walk('data'):
    for file in files:
        if file.endswith('.wav'):
            waveform, sample_rate = torchaudio.load(os.path.join(root, file))
            print(f'{file}: {sample_rate}Hz, {waveform.shape}')
"
```

## Synthetic Data Detection

Our verification includes checks for synthetic data indicators:

### 1. Uniform Noise Detection
- **Method**: Standard deviation analysis
- **Threshold**: std < 0.01 indicates artificial noise
- **Action**: Flag suspicious files

### 2. Repetitive Pattern Detection
- **Method**: Autocorrelation analysis
- **Threshold**: autocorr > 0.9 indicates repetition
- **Action**: Flag suspicious files

### 3. Artificial Spectral Analysis
- **Method**: FFT magnitude uniformity
- **Threshold**: std(magnitude) < 0.1 indicates artificial spectrum
- **Action**: Flag suspicious files

### 4. Duration Analysis
- **Method**: File duration distribution
- **Expected**: Natural variation in real recordings
- **Action**: Flag unnaturally uniform durations

## Data Integrity Assurance

### Checksums

All downloaded files are verified against SHA256 checksums:

```python
# Example checksum verification
import hashlib

def verify_checksum(filepath, expected_checksum):
    with open(filepath, 'rb') as f:
        actual_checksum = hashlib.sha256(f.read()).hexdigest()
    return actual_checksum == expected_checksum
```

### Size Validation

Files are validated against expected sizes:

```python
# Example size validation
def verify_file_size(filepath, expected_size_mb):
    actual_size_mb = filepath.stat().st_size / (1024 * 1024)
    return abs(actual_size_mb - expected_size_mb) < 10  # 10MB tolerance
```

### Audio Quality Checks

Each audio file is validated:

```python
# Example audio validation
def validate_audio_file(filepath):
    waveform, sample_rate = torchaudio.load(filepath)
    
    # Check sample rate
    if sample_rate != 16000:
        return False
    
    # Check duration
    duration = waveform.shape[1] / sample_rate
    if not (1.0 <= duration <= 60.0):
        return False
    
    # Check for silence/corruption
    if torch.all(waveform == 0) or torch.isnan(waveform).any():
        return False
    
    return True
```

## Dataset Statistics

### VoiceBank+DEMAND
- **Total Files**: 11,572
- **Clean Speech**: 5,764 files
- **Noisy Speech**: 5,808 files
- **Total Size**: ~2.4 GB
- **Average Duration**: 4.2 seconds
- **Sample Rate**: 16 kHz

### LibriSpeech (dev-clean)
- **Total Files**: 2,703
- **Total Duration**: ~5.4 hours
- **Total Size**: ~337 MB
- **Average Duration**: 7.2 seconds
- **Sample Rate**: 16 kHz

### Microsoft DNS
- **Total Files**: ~65,000
- **Total Duration**: ~18 hours
- **Total Size**: ~890 MB
- **Average Duration**: 1.0 seconds
- **Sample Rate**: 16 kHz

### Freesound Noise
- **Total Files**: ~1,000
- **Total Duration**: ~2.8 hours
- **Total Size**: ~156 MB
- **Average Duration**: 10.1 seconds
- **Sample Rate**: 16 kHz

## Data Usage

### Training Data
- **Clean Speech**: VoiceBank+DEMAND clean files
- **Noise**: DEMAND, Microsoft DNS, Freesound noise
- **Mixing**: Real noise added to real speech
- **No Synthetic Generation**: All mixing uses real recordings

### Validation Data
- **Clean Speech**: VoiceBank+DEMAND test clean files
- **Noise**: Same sources as training
- **Evaluation**: Real noisy speech enhancement

### Test Data
- **Clean Speech**: LibriSpeech dev-clean
- **Noise**: Microsoft DNS, Freesound
- **Evaluation**: Cross-dataset generalization

## Citation Requirements

When using these datasets, please cite:

```bibtex
@article{voicebank_demand_2018,
  title={The VoiceBank corpus: Design, collection and data analysis of a phonetically balanced corpus of speech},
  author={Veaux, Christophe and Yamagishi, Junichi and King, Simon},
  journal={Speech Communication},
  volume={98},
  pages={1--13},
  year={2018}
}

@article{librispeech_2015,
  title={LibriSpeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  journal={arXiv preprint arXiv:1412.5567},
  year={2015}
}

@article{dns_challenge_2020,
  title={The INTERSPEECH 2020 deep noise suppression challenge: Datasets, subjective testing framework, and challenge results},
  author={Reddy, Chandan KA and Gopal, Vishak and Cutler, Ross and Beyrami, Ebrahim and Cheng, Roger and Dubey, Harish and Matusevych, Sergiy and Aichner, Robert and Aazami, Ashkan and Braun, Sebastian and others},
  journal={arXiv preprint arXiv:2001.08662},
  year={2020}
}
```

## Quality Assurance

### Automated Checks
- **Daily**: Dataset integrity verification
- **Weekly**: Synthetic data detection
- **Monthly**: Full dataset re-verification

### Manual Reviews
- **Quarterly**: Random sample audio review
- **Annually**: Full dataset audit

### Reporting
- **Verification Reports**: JSON format with detailed results
- **Quality Metrics**: Audio quality statistics
- **Synthetic Detection**: Flagged files and patterns

## Contact

For questions about data sources or verification:

- **Dataset Issues**: Open an issue on GitHub
- **Verification Problems**: Check logs in `verification_report.json`
- **Manual Download**: Follow instructions in this document

---

**⚠️ Important**: This project uses **real datasets only**. No synthetic data generation is performed. All results are reproducible with the provided datasets and verification procedures. 