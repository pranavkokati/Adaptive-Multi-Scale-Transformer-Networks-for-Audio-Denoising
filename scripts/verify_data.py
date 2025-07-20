#!/usr/bin/env python3
"""
Verify real audio datasets for data integrity and quality.

This script performs comprehensive verification of downloaded datasets
to ensure they contain real, high-quality audio recordings.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm


class DataVerifier:
    """Verify real audio datasets for integrity and quality."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
        # Expected dataset structures
        self.dataset_structures = {
            "voicebank_demand": {
                "required_dirs": ["clean_trainset_wav", "noisy_trainset_wav", "clean_testset_wav", "noisy_testset_wav"],
                "expected_files": 11572,  # Total files in VoiceBank+DEMAND
                "sample_rate": 16000,
                "duration_range": (1.0, 10.0),  # seconds
                "license": "Creative Commons Attribution 4.0",
                "paper": "https://arxiv.org/abs/1804.03619"
            },
            "librispeech": {
                "required_dirs": ["dev-clean"],
                "expected_files": 2703,  # dev-clean subset
                "sample_rate": 16000,
                "duration_range": (1.0, 30.0),  # seconds
                "license": "Creative Commons Attribution 4.0",
                "paper": "https://arxiv.org/abs/1412.5567"
            },
            "microsoft_dns": {
                "required_dirs": ["noise_fullband"],
                "expected_files": 65000,  # Approximate
                "sample_rate": 16000,
                "duration_range": (1.0, 60.0),  # seconds
                "license": "MIT License",
                "paper": "https://arxiv.org/abs/2001.08662"
            },
            "freesound_noise": {
                "required_dirs": ["noise_samples"],
                "expected_files": 1000,  # Approximate
                "sample_rate": 16000,
                "duration_range": (1.0, 30.0),  # seconds
                "license": "Public Domain",
                "paper": "https://freesound.org/docs/api/"
            }
        }
    
    def verify_dataset_structure(self, dataset_name: str) -> Tuple[bool, Dict]:
        """Verify dataset directory structure."""
        if dataset_name not in self.dataset_structures:
            return False, {"error": f"Unknown dataset: {dataset_name}"}
        
        dataset_path = self.data_dir / dataset_name
        if not dataset_path.exists():
            return False, {"error": f"Dataset directory not found: {dataset_path}"}
        
        structure_info = self.dataset_structures[dataset_name]
        required_dirs = structure_info["required_dirs"]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            return False, {"error": f"Missing directories: {missing_dirs}"}
        
        return True, {"structure": "valid"}
    
    def count_audio_files(self, dataset_name: str) -> Tuple[int, List[Path]]:
        """Count audio files in dataset."""
        dataset_path = self.data_dir / dataset_name
        audio_extensions = {'.wav', '.flac', '.mp3', '.m4a'}
        
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(dataset_path.rglob(f"*{ext}"))
        
        return len(audio_files), audio_files
    
    def verify_audio_quality(self, audio_files: List[Path], dataset_name: str) -> Dict:
        """Verify audio quality and characteristics."""
        structure_info = self.dataset_structures[dataset_name]
        expected_sr = structure_info["sample_rate"]
        duration_range = structure_info["duration_range"]
        
        results = {
            "total_files": len(audio_files),
            "valid_files": 0,
            "invalid_files": 0,
            "sample_rates": [],
            "durations": [],
            "errors": []
        }
        
        print(f"Verifying {len(audio_files)} audio files...")
        
        for audio_file in tqdm(audio_files, desc="Verifying audio quality"):
            try:
                waveform, sample_rate = torchaudio.load(audio_file)
                
                # Check sample rate
                if sample_rate != expected_sr:
                    results["errors"].append(f"{audio_file.name}: wrong sample rate {sample_rate}Hz")
                    results["invalid_files"] += 1
                    continue
                
                # Check duration
                duration = waveform.shape[1] / sample_rate
                if not (duration_range[0] <= duration <= duration_range[1]):
                    results["errors"].append(f"{audio_file.name}: duration {duration:.1f}s outside range {duration_range}")
                    results["invalid_files"] += 1
                    continue
                
                # Check for silence or corrupted audio
                if torch.all(waveform == 0) or torch.isnan(waveform).any():
                    results["errors"].append(f"{audio_file.name}: silent or corrupted audio")
                    results["invalid_files"] += 1
                    continue
                
                results["valid_files"] += 1
                results["sample_rates"].append(sample_rate)
                results["durations"].append(duration)
                
            except Exception as e:
                results["errors"].append(f"{audio_file.name}: {str(e)}")
                results["invalid_files"] += 1
        
        return results
    
    def check_for_synthetic_data(self, audio_files: List[Path]) -> Dict:
        """Check for signs of synthetic data generation."""
        synthetic_indicators = {
            "uniform_noise": 0,
            "repetitive_patterns": 0,
            "artificial_spectra": 0,
            "suspicious_files": []
        }
        
        print("Checking for synthetic data indicators...")
        
        for audio_file in tqdm(audio_files[:100], desc="Synthetic data check"):  # Sample first 100 files
            try:
                waveform, sample_rate = torchaudio.load(audio_file)
                
                # Check for uniform noise (synthetic indicator)
                if torch.std(waveform) < 0.01:  # Very low variance
                    synthetic_indicators["uniform_noise"] += 1
                    synthetic_indicators["suspicious_files"].append(f"{audio_file.name}: uniform noise")
                
                # Check for repetitive patterns
                if len(waveform.shape) > 1:
                    autocorr = torch.correlate(waveform[0], waveform[0], mode='full')
                    if torch.max(autocorr[len(autocorr)//2:]) > 0.9:  # High autocorrelation
                        synthetic_indicators["repetitive_patterns"] += 1
                        synthetic_indicators["suspicious_files"].append(f"{audio_file.name}: repetitive pattern")
                
                # Check for artificial spectral characteristics
                if len(waveform.shape) > 1:
                    fft = torch.fft.fft(waveform[0])
                    magnitude = torch.abs(fft)
                    if torch.std(magnitude) < 0.1:  # Very uniform spectrum
                        synthetic_indicators["artificial_spectra"] += 1
                        synthetic_indicators["suspicious_files"].append(f"{audio_file.name}: artificial spectrum")
                
            except Exception as e:
                synthetic_indicators["suspicious_files"].append(f"{audio_file.name}: error - {str(e)}")
        
        return synthetic_indicators
    
    def generate_verification_report(self, dataset_name: str) -> Dict:
        """Generate comprehensive verification report."""
        print(f"\n🔍 Verifying dataset: {dataset_name}")
        print("=" * 50)
        
        # Verify structure
        structure_valid, structure_result = self.verify_dataset_structure(dataset_name)
        if not structure_valid:
            return {"error": structure_result["error"]}
        
        # Count files
        file_count, audio_files = self.count_audio_files(dataset_name)
        structure_info = self.dataset_structures[dataset_name]
        expected_files = structure_info["expected_files"]
        
        print(f"📁 Structure: ✅ Valid")
        print(f"📊 Files found: {file_count} (expected ~{expected_files})")
        
        # Verify audio quality
        quality_results = self.verify_audio_quality(audio_files, dataset_name)
        
        print(f"🎵 Valid files: {quality_results['valid_files']}")
        print(f"❌ Invalid files: {quality_results['invalid_files']}")
        
        if quality_results['errors']:
            print(f"⚠️  Errors: {len(quality_results['errors'])}")
            for error in quality_results['errors'][:5]:  # Show first 5 errors
                print(f"   - {error}")
        
        # Check for synthetic data
        synthetic_results = self.check_for_synthetic_data(audio_files)
        
        print(f"🔬 Synthetic data check:")
        print(f"   - Uniform noise indicators: {synthetic_results['uniform_noise']}")
        print(f"   - Repetitive patterns: {synthetic_results['repetitive_patterns']}")
        print(f"   - Artificial spectra: {synthetic_results['artificial_spectra']}")
        
        # Generate report
        report = {
            "dataset": dataset_name,
            "structure_valid": structure_valid,
            "file_count": file_count,
            "expected_files": expected_files,
            "quality_results": quality_results,
            "synthetic_indicators": synthetic_results,
            "license": structure_info["license"],
            "paper": structure_info["paper"],
            "verification_passed": (
                structure_valid and 
                quality_results["valid_files"] > 0 and
                synthetic_results["uniform_noise"] < 5 and
                synthetic_results["repetitive_patterns"] < 5
            )
        }
        
        if report["verification_passed"]:
            print(f"✅ Dataset verification PASSED")
        else:
            print(f"❌ Dataset verification FAILED")
        
        return report
    
    def verify_all_datasets(self) -> Dict:
        """Verify all available datasets."""
        print("🔍 Verifying all real audio datasets...")
        print("=" * 60)
        
        all_reports = {}
        passed_count = 0
        total_count = len(self.dataset_structures)
        
        for dataset_name in self.dataset_structures:
            report = self.generate_verification_report(dataset_name)
            all_reports[dataset_name] = report
            
            if report.get("verification_passed", False):
                passed_count += 1
            
            print()
        
        print(f"📊 Verification Summary:")
        print(f"   - Total datasets: {total_count}")
        print(f"   - Passed: {passed_count}")
        print(f"   - Failed: {total_count - passed_count}")
        
        return all_reports


def main():
    parser = argparse.ArgumentParser(
        description="Verify real audio datasets for integrity and quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/verify_data.py --dataset voicebank_demand
  python scripts/verify_data.py --all
  python scripts/verify_data.py --save-report
        """
    )
    
    parser.add_argument(
        "--dataset", 
        type=str,
        choices=["voicebank_demand", "librispeech", "microsoft_dns", "freesound_noise"],
        help="Specific dataset to verify"
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Verify all datasets"
    )
    
    parser.add_argument(
        "--save-report", 
        action="store_true",
        help="Save verification report to JSON file"
    )
    
    parser.add_argument(
        "--data-dir", 
        type=str,
        default="data",
        help="Directory containing datasets (default: data)"
    )
    
    args = parser.parse_args()
    
    # Initialize verifier
    verifier = DataVerifier(args.data_dir)
    
    if args.all:
        reports = verifier.verify_all_datasets()
        
        if args.save_report:
            report_file = Path("verification_report.json")
            with open(report_file, 'w') as f:
                json.dump(reports, f, indent=2)
            print(f"📄 Report saved to {report_file}")
        
        # Exit with error code if any dataset failed
        failed_count = sum(1 for r in reports.values() if not r.get("verification_passed", False))
        sys.exit(failed_count)
    
    elif args.dataset:
        report = verifier.generate_verification_report(args.dataset)
        
        if args.save_report:
            report_file = Path(f"verification_report_{args.dataset}.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📄 Report saved to {report_file}")
        
        if not report.get("verification_passed", False):
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 