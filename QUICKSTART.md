# Quick Start Guide

Get started with the kinetics_nn pipeline in 5 minutes!

## Prerequisites Check

Before starting, ensure you have:

1. ✓ Python 3.8 or higher installed
2. ✓ Microsoft Excel installed and working
3. ✓ The Excel workbook: `PFR-CSTR-cascade_Calculation-macro.xlsm`

## Step 1: Setup (2 minutes)

```bash
# Navigate to project directory
cd ~/kinetics_nn

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Update Workbook Path (30 seconds)

Edit the workbook path in your quickstart script:

**For macOS/Linux** - Edit `quickstart.sh`:
```bash
WORKBOOK="/mnt/data/PFR-CSTR-cascade_Calculation-macro.xlsm"
```

**For Windows** - Edit `quickstart.bat`:
```batch
set WORKBOOK=C:\path\to\PFR-CSTR-cascade_Calculation-macro.xlsm
```

Replace with the actual path to your workbook.

## Step 3: Run Smoke Test (5-10 minutes)

### Option A: Automated Script

**macOS/Linux**:
```bash
./quickstart.sh
```

**Windows**:
```batch
quickstart.bat
```

### Option B: Manual Commands

```bash
# Test Excel connection
python -m src.excel_driver /path/to/workbook.xlsm

# Generate small dataset (50 runs, ~2 minutes)
python -m src.dataset_builder \
    --workbook /path/to/workbook.xlsm \
    --n_runs 50 \
    --save_dir ./data

# Preprocess data (~5 seconds)
python -m src.preprocess \
    --data_path ./data/dataset.npz \
    --save_dir ./data

# Train model (10 epochs, ~2 minutes)
python -m src.train \
    --data_dir ./data \
    --epochs 10 \
    --batch_size 8 \
    --save_dir ./models

# Make prediction (~10 seconds)
python -m src.predict \
    --excel \
    --workbook /path/to/workbook.xlsm \
    --reactor PFR \
    --A0 0.2 --B0 0.15 --I0 0.05 \
    --k1 1e-10 --k2 0 \
    --model_path ./models/best_model.pt \
    --scaler_path ./data/scaler.json \
    --config_path ./models/training_config.json
```

## Step 4: Check Results

After the smoke test completes:

1. **View training curves**:
   ```bash
   open models/training_curves.png
   ```

2. **View confusion matrix**:
   ```bash
   open models/confusion_matrix.png
   ```

3. **Check prediction output** - should see something like:
   ```
   Mechanism Predictions (ranked by probability):
   ============================================================

   1. Equation: A + B -> C
      Type: second_order
      Probability: 0.9234 (92.34%)
      Class ID: 0
   ```

## Step 5: Generate Production Dataset (Optional)

For better model performance, generate a larger dataset:

```bash
# Generate 2000 runs (~30-60 minutes)
python -m src.dataset_builder \
    --workbook /path/to/workbook.xlsm \
    --n_runs 2000 \
    --save_dir ./data

# Preprocess
python -m src.preprocess \
    --data_path ./data/dataset.npz \
    --save_dir ./data

# Train with more epochs (~10-30 minutes on CPU)
python -m src.train \
    --data_dir ./data \
    --epochs 50 \
    --save_dir ./models \
    --device cpu
```

### Use GPU for Faster Training (if available)

**Apple Silicon (M1/M2)**:
```bash
python -m src.train --data_dir ./data --epochs 50 --device mps
```

**NVIDIA GPU**:
```bash
python -m src.train --data_dir ./data --epochs 50 --device cuda
```

## Common Use Cases

### Predict from New Parameters

```bash
python -m src.predict \
    --excel \
    --workbook /path/to/workbook.xlsm \
    --reactor PFR \
    --A0 0.3 --B0 0.2 --I0 0.1 \
    --k1 2e-10 --k2 1e-10 \
    --show_all
```

### Change Confidence Threshold

```bash
python -m src.predict \
    --excel \
    --workbook /path/to/workbook.xlsm \
    --reactor CSTR \
    --A0 0.2 --B0 0.15 --I0 0.05 \
    --k1 0 --k2 5e-10 \
    --threshold 0.01  # Lower threshold shows more predictions
```

### Get JSON Output

```bash
python -m src.predict \
    --excel \
    --workbook /path/to/workbook.xlsm \
    --reactor PFR \
    --A0 0.2 --B0 0.1 --I0 0.05 \
    --k1 1e-10 --k2 0 \
    --json_output > prediction.json
```

## Troubleshooting

### Excel won't open

**Problem**: `xlwings` cannot connect to Excel

**Solution**:
1. Ensure Excel is open and working
2. Try with visible Excel: add `--visible` flag
3. On macOS: Check System Preferences → Security & Privacy

### Import errors

**Problem**: `ModuleNotFoundError`

**Solution**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Out of memory

**Problem**: System runs out of memory during training

**Solution**:
```bash
# Reduce batch size
python -m src.train --batch_size 8

# Or reduce dataset size
python -m src.dataset_builder --n_runs 500
```

### Low accuracy

**Problem**: Model accuracy is below 80%

**Solution**:
1. Generate more data: `--n_runs 2000` or higher
2. Train longer: `--epochs 100`
3. Try different model: `--model_type gru`

## Next Steps

1. Read [README.md](README.md) for detailed documentation
2. Review [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for architecture overview
3. Check example configs in `configs/`
4. Customize parameters for your use case

## Support

For issues or questions:
1. Check README.md troubleshooting section
2. Review error messages carefully
3. Test with smaller datasets first
4. Contact the author

## Quick Reference

### File Locations

- **Models**: `./models/best_model.pt`
- **Dataset**: `./data/dataset.npz`
- **Scaler**: `./data/scaler.json`
- **Plots**: `./models/*.png`

### Important Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_runs` | 1000 | Number of simulation runs |
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--threshold` | 0.05 | Prediction confidence threshold |
| `--device` | cpu | Device (cpu/cuda/mps) |

### Mechanism Classes

- **Class 0**: A + B → C (k1>0, k2=0)
- **Class 1**: 2A → D (k1=0, k2>0)
- **Class 2**: Both reactions (k1>0, k2>0)

---

**Ready to start?** Run the smoke test and see the magic happen! 🚀
