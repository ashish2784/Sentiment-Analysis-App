#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Install Python dependencies
pip install -r requirements.txt

# Run the model training script to generate the .pkl files
python train_model.py