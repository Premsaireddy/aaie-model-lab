#!/usr/bin/env python3
"""
Check progress of background Phi2 evaluation
"""

import os
import time
from pathlib import Path

def check_progress():
    """Check if Phi2 evaluation is running and show progress"""
    print("🔍 Checking Phi2 Evaluation Progress")
    print("=" * 40)
    
    # Check for result files
    result_locations = [
        "/workspace/evaluation_results",
        "/workspace/quick_test_results"
    ]
    
    for location in result_locations:
        result_dir = Path(location)
        if result_dir.exists():
            files = list(result_dir.glob("*"))
            if files:
                print(f"📁 Found results in {location}:")
                for file in files:
                    size = file.stat().st_size if file.is_file() else 0
                    modified = time.ctime(file.stat().st_mtime)
                    print(f"  📄 {file.name} ({size} bytes, modified: {modified})")
            else:
                print(f"📂 Empty directory: {location}")
        else:
            print(f"❌ Directory not found: {location}")
    
    # Check for running processes
    print(f"\n🔄 Checking for running Python processes...")
    try:
        import subprocess
        result = subprocess.run(['pgrep', '-f', 'python.*phi2'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"✅ Found {len(pids)} running Phi2 processes:")
            for pid in pids:
                print(f"  PID: {pid}")
        else:
            print("❌ No Phi2 processes currently running")
    except Exception as e:
        print(f"⚠️  Could not check processes: {e}")
    
    # Check system resources
    print(f"\n💻 System Resources:")
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        print(f"  CPU Usage: {cpu_percent:.1f}%")
        print(f"  Memory Usage: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
        
        # Check GPU if available
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  GPU Available: Yes ({torch.cuda.get_device_name()})")
                print(f"  GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB / {torch.cuda.memory_reserved()/1024**3:.1f}GB")
            else:
                print(f"  GPU Available: No")
        except:
            print(f"  GPU Status: Unknown")
            
    except ImportError:
        print("  System monitoring unavailable (psutil not installed)")
    except Exception as e:
        print(f"  Error checking resources: {e}")

def main():
    check_progress()
    
    print(f"\n💡 Available Commands:")
    print("  python3 mock_phi2_test.py      # Demo with mock responses")
    print("  python3 quick_phi2_test.py     # Quick test with real model")
    print("  python3 run_phi2_tests.py      # Full evaluation")
    print("  python3 results_analyzer.py    # Analyze results")
    print("  python3 check_phi2_progress.py # Check this status")

if __name__ == "__main__":
    main()