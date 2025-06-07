#!/bin/bash

# Black Box Challenge - Your Implementation
# This script should take three parameters and output the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Check if model file exists, if not download it
if [ ! -f "production_model.pkl" ]; then
    echo "Model file not found. Downloading from GitHub..."
    
    # Try to download from GitHub releases
    MODEL_URL="https://github.com/zsiegel92/top-coder-8090/releases/download/v1.0/production_model.pkl"
    
    if command -v curl &> /dev/null; then
        curl -L -o production_model.pkl "$MODEL_URL"
    elif command -v wget &> /dev/null; then
        wget -O production_model.pkl "$MODEL_URL"
    else
        echo "Error: Neither curl nor wget found. Cannot download model file."
        echo "Please install curl or wget, or manually download the model file."
        exit 1
    fi
    
    # Check if download was successful
    if [ ! -f "production_model.pkl" ]; then
        echo "Failed to download model file. Training a new model instead..."
        python -c "from model import train_production_model; train_production_model()"
    else
        echo "Model downloaded successfully."
    fi
fi

python main.py "$1" "$2" "$3"