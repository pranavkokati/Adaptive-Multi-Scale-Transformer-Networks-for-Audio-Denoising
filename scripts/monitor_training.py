#!/usr/bin/env python3

import time
import subprocess
import psutil
import os
from pathlib import Path
import json
import torch


def get_training_status():
    """Check if training process is running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'train_librispeech.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pid = int(result.stdout.strip())
            return True, pid
        return False, None
    except:
        return False, None


def get_process_info(pid):
    """Get process information"""
    try:
        process = psutil.Process(pid)
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'status': process.status(),
            'create_time': process.create_time()
        }
    except:
        return None


def check_checkpoints():
    """Check for saved checkpoints"""
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    for file in checkpoint_dir.glob("*.pt"):
        stat = file.stat()
        checkpoints.append({
            'name': file.name,
            'size_mb': stat.st_size / 1024 / 1024,
            'modified': time.ctime(stat.st_mtime)
        })
    return checkpoints


def check_logs():
    """Check training logs"""
    log_dir = Path("logs")
    if not log_dir.exists():
        return []
    
    logs = []
    for file in log_dir.glob("*.log"):
        stat = file.stat()
        logs.append({
            'name': file.name,
            'size_kb': stat.st_size / 1024,
            'modified': time.ctime(stat.st_mtime)
        })
    return logs


def main():
    print("🔍 ADAPTIVE MULTI-SCALE TRANSFORMER - TRAINING MONITOR")
    print("="*60)
    
    monitor_count = 0
    
    while True:
        monitor_count += 1
        print(f"\n📊 MONITOR UPDATE #{monitor_count} - {time.strftime('%H:%M:%S')}")
        print("-" * 50)
        
        # Check training status
        is_running, pid = get_training_status()
        
        if is_running:
            print(f"✅ Training Status: ACTIVE (PID: {pid})")
            
            # Get process info
            proc_info = get_process_info(pid)
            if proc_info:
                print(f"🖥️  CPU Usage: {proc_info['cpu_percent']:.1f}%")
                print(f"💾 Memory Usage: {proc_info['memory_mb']:.1f} MB")
                print(f"⏱️  Running Time: {time.time() - proc_info['create_time']:.0f} seconds")
        else:
            print("❌ Training Status: NOT RUNNING")
        
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"\n🖥️  SYSTEM RESOURCES:")
        print(f"   CPU: {cpu_percent:.1f}%")
        print(f"   Memory: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
        
        # Check GPU if available
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
        
        # Check checkpoints
        checkpoints = check_checkpoints()
        print(f"\n💾 CHECKPOINTS ({len(checkpoints)} found):")
        for cp in checkpoints[-3:]:  # Show last 3
            print(f"   📁 {cp['name']} ({cp['size_mb']:.1f}MB) - {cp['modified']}")
        
        # Check logs
        logs = check_logs()
        print(f"\n📝 LOGS ({len(logs)} found):")
        for log in logs[-2:]:  # Show last 2
            print(f"   📄 {log['name']} ({log['size_kb']:.1f}KB) - {log['modified']}")
        
        # Dataset status
        librispeech_path = Path("datasets/librispeech/LibriSpeech/dev-clean")
        if librispeech_path.exists():
            audio_files = list(librispeech_path.glob("**/*.flac"))
            print(f"\n🎵 DATASET STATUS:")
            print(f"   ✅ LibriSpeech: {len(audio_files)} real audio files")
        else:
            print(f"\n🎵 DATASET STATUS:")
            print(f"   ⏳ LibriSpeech: Downloading...")
        
        # Project structure check
        required_dirs = ['src', 'scripts', 'configs', 'datasets', 'checkpoints', 'logs', 'outputs']
        missing_dirs = [d for d in required_dirs if not Path(d).exists()]
        
        print(f"\n📁 PROJECT STRUCTURE:")
        if not missing_dirs:
            print(f"   ✅ All directories present")
        else:
            print(f"   ⚠️  Missing: {', '.join(missing_dirs)}")
        
        print(f"\n🎯 NEXT STEPS:")
        if is_running:
            print(f"   ⏳ Training in progress - model will be ready soon")
            print(f"   📈 Monitor logs for training metrics")
            print(f"   🔄 Checkpoints will be saved automatically")
        else:
            if checkpoints:
                print(f"   ✅ Training completed - model ready for inference")
                print(f"   🚀 Run: python3 scripts/inference_real.py")
                print(f"   📊 Run: python3 scripts/evaluate_real.py")
            else:
                print(f"   🔄 Start training: python3 scripts/train_librispeech.py")
        
        print(f"\n" + "="*60)
        print(f"Press Ctrl+C to stop monitoring")
        
        try:
            time.sleep(10)  # Update every 10 seconds
        except KeyboardInterrupt:
            print(f"\n👋 Monitoring stopped")
            break


if __name__ == "__main__":
    main()
