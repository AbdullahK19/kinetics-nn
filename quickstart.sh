#!/bin/bash
# Quick start script for kinetics_nn pipeline
# This script runs a complete smoke test with 50 runs

set -e  # Exit on error

echo "================================================"
echo "Kinetics NN - Quick Start Smoke Test"
echo "================================================"
echo ""

# Configuration
WORKBOOK="/mnt/data/PFR-CSTR-cascade_Calculation-macro.xlsm"
N_RUNS=50
EPOCHS=10
BATCH_SIZE=8

# Check if workbook exists
if [ ! -f "$WORKBOOK" ]; then
    echo "Error: Excel workbook not found at: $WORKBOOK"
    echo "Please update the WORKBOOK variable in this script with the correct path."
    exit 1
fi

echo "Step 1: Testing Excel driver..."
python -m src.excel_driver "$WORKBOOK"
echo ""

echo "Step 2: Generating dataset ($N_RUNS runs)..."
python -m src.dataset_builder \
    --workbook "$WORKBOOK" \
    --n_runs $N_RUNS \
    --save_dir ./data \
    --seed 42
echo ""

echo "Step 3: Preprocessing data..."
python -m src.preprocess \
    --data_path ./data/dataset.npz \
    --save_dir ./data \
    --seed 42
echo ""

echo "Step 4: Training model ($EPOCHS epochs)..."
python -m src.train \
    --data_dir ./data \
    --model_type cnn \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr 0.001 \
    --save_dir ./models \
    --device cpu \
    --seed 42
echo ""

echo "Step 5: Testing prediction..."
python -m src.predict \
    --excel \
    --workbook "$WORKBOOK" \
    --reactor PFR \
    --A0 0.2 \
    --B0 0.15 \
    --I0 0.05 \
    --k1 1e-10 \
    --k2 0 \
    --threshold 0.05 \
    --show_all \
    --model_path ./models/best_model.pt \
    --scaler_path ./data/scaler.json \
    --config_path ./models/training_config.json
echo ""

echo "================================================"
echo "Smoke test complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  - Check training curves: open models/training_curves.png"
echo "  - Check confusion matrix: open models/confusion_matrix.png"
echo "  - Generate full dataset: python -m src.dataset_builder --n_runs 2000"
echo "  - Train full model: python -m src.train --epochs 50"
echo ""
