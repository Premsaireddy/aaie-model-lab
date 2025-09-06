#!/usr/bin/env python3
"""
Test script to verify the ZSL/FSL experiment setup
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    required_packages = [
        ('openai', 'OpenAI API'),
        ('pandas', 'Data processing'),
        ('numpy', 'Numerical computing'),
        ('sklearn', 'Machine learning metrics'),
        ('matplotlib', 'Plotting'),
        ('seaborn', 'Statistical visualization'),
        ('plotly', 'Interactive visualization'),
        ('openpyxl', 'Excel export'),
    ]
    
    failed = []
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package:15} - {description}")
        except ImportError:
            print(f"  ✗ {package:15} - {description}")
            failed.append(package)
    
    if failed:
        print(f"\n⚠ Missing packages: {', '.join(failed)}")
        print("  Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All required packages installed")
    return True


def test_api_key():
    """Test if OpenAI API key is configured"""
    print("\nTesting API key configuration...")
    
    # Check environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Check .env file
    env_file = Path(".env")
    if not api_key and env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
        except ImportError:
            pass
    
    if api_key:
        # Mask the key for security
        masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 11 else "***"
        print(f"  ✓ API key found: {masked_key}")
        
        # Basic validation
        if not api_key.startswith("sk-"):
            print("  ⚠ Warning: API key doesn't start with 'sk-'")
        
        return True
    else:
        print("  ✗ API key not found")
        print("  Set it using: export OPENAI_API_KEY='your-key-here'")
        print("  Or create a .env file with: OPENAI_API_KEY=your-key-here")
        return False


def test_files():
    """Test if required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'openai_zsl_fsl_pipeline.py',
        'visualization_analysis.py',
        'config.py',
        'run_experiment.py',
        'create_sample_data.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file}")
            missing.append(file)
    
    if missing:
        print(f"\n⚠ Missing files: {', '.join(missing)}")
        return False
    
    print("\n✓ All required files present")
    return True


def test_data():
    """Test if sample data exists"""
    print("\nTesting data files...")
    
    data_files = [
        'accounting.json',
        'engineering.json',
        'it.json',
        'psychology.json',
        'teaching.json'
    ]
    
    existing = []
    missing = []
    
    for file in data_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
            existing.append(file)
        else:
            print(f"  ✗ {file}")
            missing.append(file)
    
    if missing:
        print(f"\n⚠ Missing data files: {', '.join(missing)}")
        print("  Run: python create_sample_data.py")
        if existing:
            print(f"  Available files: {', '.join(existing)}")
            return True  # Partial success
        return False
    
    print("\n✓ All data files present")
    return True


def test_directories():
    """Test if output directories exist"""
    print("\nTesting directories...")
    
    directories = [
        'outputs',
        'outputs/visualizations',
        'outputs/reports'
    ]
    
    for directory in directories:
        path = Path(directory)
        if path.exists():
            print(f"  ✓ {directory}/")
        else:
            print(f"  ✗ {directory}/ (will be created automatically)")
    
    return True


def test_pipeline():
    """Test if the pipeline can be imported and initialized"""
    print("\nTesting pipeline initialization...")
    
    try:
        from openai_zsl_fsl_pipeline import OpenAIZSLFSLPipeline
        print("  ✓ Pipeline module imported")
        
        # Try to initialize (will fail without API key, but that's ok)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                pipeline = OpenAIZSLFSLPipeline(api_key=api_key)
                print("  ✓ Pipeline initialized successfully")
                return True
            except Exception as e:
                print(f"  ⚠ Pipeline initialization failed: {e}")
                return False
        else:
            print("  ⚠ Skipping initialization (no API key)")
            return True
            
    except ImportError as e:
        print(f"  ✗ Failed to import pipeline: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("ZSL/FSL Experiment Setup Test")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("API Key", test_api_key),
        ("Files", test_files),
        ("Data", test_data),
        ("Directories", test_directories),
        ("Pipeline", test_pipeline)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with error: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:15} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ Setup is complete and ready to run!")
        print("\nNext step: python run_experiment.py --visualize --dashboard")
        return 0
    elif results.get("Imports") and results.get("Files"):
        print("\n⚠ Setup is partially complete")
        print("  Some optional components are missing but core functionality should work")
        return 0
    else:
        print("\n✗ Setup is incomplete")
        print("  Please address the issues above before running experiments")
        return 1


if __name__ == "__main__":
    sys.exit(main())