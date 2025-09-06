#!/bin/bash

# Setup script for ZSL/FSL experiment with OpenAI GPT-4

echo "========================================="
echo "ZSL/FSL Experiment Setup"
echo "========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python 3.8 or higher is required (found $python_version)"
    exit 1
fi

echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ Pip upgraded"

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt --quiet
echo "✓ Requirements installed"

# Check for API key
echo ""
echo "Checking for OpenAI API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f ".env" ]; then
        if grep -q "OPENAI_API_KEY" .env; then
            echo "✓ API key found in .env file"
        else
            echo "⚠ Warning: OPENAI_API_KEY not found in .env file"
            echo "  Add your API key to .env file:"
            echo "  OPENAI_API_KEY=your-key-here"
        fi
    else
        echo "⚠ Warning: OPENAI_API_KEY environment variable not set"
        echo "  Set it using: export OPENAI_API_KEY='your-key-here'"
        echo "  Or create a .env file with: OPENAI_API_KEY=your-key-here"
    fi
else
    echo "✓ API key found in environment"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p outputs/visualizations outputs/reports data
echo "✓ Directories created"

# Create sample data
echo ""
echo "Creating sample data files..."
python3 create_sample_data.py
echo "✓ Sample data created"

# Display next steps
echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Set your OpenAI API key (if not already done):"
echo "   export OPENAI_API_KEY='your-key-here'"
echo ""
echo "2. Run the experiment:"
echo "   python run_experiment.py --visualize --dashboard"
echo ""
echo "3. View results in outputs/ directory"
echo ""
echo "For help:"
echo "   python run_experiment.py --help"
echo ""
echo "========================================="