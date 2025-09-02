#!/usr/bin/env python3
"""
Installation and execution script for Phi-2 Academic Analyzer
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def run_analyzer():
    """Run the Phi-2 academic analyzer"""
    print("\n" + "="*60)
    print("STARTING PHI-2 ACADEMIC ANALYZER")
    print("="*60)
    
    try:
        subprocess.check_call([sys.executable, "phi2_academic_analyzer.py"])
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running analyzer: {e}")
        return False
    except FileNotFoundError:
        print("❌ phi2_academic_analyzer.py not found!")
        return False
    return True

def main():
    """Main execution function"""
    print("🚀 Phi-2 Academic Analyzer Setup")
    print("-" * 40)
    
    # Check if data files exist
    required_files = ["psychology.json", "engineering.json"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"⚠️  Warning: Missing data files: {missing_files}")
        print("The analyzer will skip missing datasets.")
    
    # Install requirements
    if not install_requirements():
        print("❌ Installation failed. Please check error messages above.")
        return
    
    # Run analyzer
    if not run_analyzer():
        print("❌ Analysis failed. Please check error messages above.")
        return
    
    print("\n✅ Analysis completed successfully!")
    print("📊 Check 'phi2_analysis_results.json' for detailed results.")

if __name__ == "__main__":
    main()