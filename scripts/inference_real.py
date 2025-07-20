#!/usr/bin/env python3

import torch
import hydra
from omegaconf import DictConfig
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.training import InferenceEngine
from src.evaluation import EvaluationSuite
from src import set_global_seed
from src.adaptive_transformer import AdaptiveMultiScaleTransformer


def main():
    parser = argparse.ArgumentParser(description='Audio Denoising Inference - Real Data Trained Model')
    parser.add_argument('--config', type=str, default='../configs/config.yaml', help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--input', type=str, required=True, help='Input audio file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output audio file or directory')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate against clean reference')
    parser.add_argument('--reference', type=str, help='Clean reference audio for evaluation')
    
    args = parser.parse_args()
    
    config = hydra.compose(config_name=args.config.replace('.yaml', '').replace('../configs/', ''))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Model trained on REAL DATA ONLY - No synthetic data used")
    
    engine = InferenceEngine(args.checkpoint, config, device)
    
    if args.batch:
        print(f"Batch processing: {args.input} -> {args.output}")
        enhanced_files = engine.batch_enhance(args.input, args.output)
        print(f"Enhanced {len(enhanced_files)} files using real-data-trained model")
    else:
        print(f"Processing: {args.input} -> {args.output}")
        enhanced_path = engine.enhance_file(args.input, args.output)
        print(f"Enhanced audio saved to: {enhanced_path}")
        
        if args.evaluate and args.reference:
            import torchaudio
            
            enhanced_audio, _ = torchaudio.load(enhanced_path)
            reference_audio, _ = torchaudio.load(args.reference)
            
            evaluator = EvaluationSuite(config.data.sample_rate)
            metrics = evaluator.audio_metrics.compute_all_metrics(
                reference_audio.squeeze(), enhanced_audio.squeeze()
            )
            
            print("\nEvaluation Results:")
            print(f"PESQ: {metrics['pesq']:.3f}")
            print(f"STOI: {metrics['stoi']:.3f}")
            print(f"SI-SDR: {metrics['si_sdr']:.2f} dB")
            print(f"SNR: {metrics['snr']:.2f} dB")


if __name__ == "__main__":
    set_global_seed(42)
    main()
