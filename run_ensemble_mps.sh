#!/bin/bash
# Runs 3 ensemble models sequentially on MPS (Apple GPU).
# Sequential = one model at a time → GPU never overloaded → laptop stays cool.
# Logs: ensemble_1.log, ensemble_2.log, ensemble_3.log

ARGS="--model_type rescnn --epochs 300
  --hidden_channels 64 128 256 --fc_hidden 256 --blocks_per_stage 2
  --dropout 0.25 --lr 0.0003 --weight_decay 0.001 --batch_size 64
  --optimizer adamw --scheduler cosine --label_smoothing 0.1
  --class_weights --augment --patience 80 --focal --focal_gamma 2.0
  --device mps"

cd /Users/abdullahkashif/kinetics_nn

echo "=== MODEL 1 (seed 42) START $(date) ===" > ensemble_1.log
PYTHONPATH=src python3 src/train.py $ARGS --seed 42  --save_dir models/ensemble_1 >> ensemble_1.log 2>&1
echo "=== MODEL 1 DONE $(date) ===" >> ensemble_1.log

echo "=== MODEL 2 (seed 123) START $(date) ===" > ensemble_2.log
PYTHONPATH=src python3 src/train.py $ARGS --seed 123 --save_dir models/ensemble_2 >> ensemble_2.log 2>&1
echo "=== MODEL 2 DONE $(date) ===" >> ensemble_2.log

echo "=== MODEL 3 (seed 456) START $(date) ===" > ensemble_3.log
PYTHONPATH=src python3 src/train.py $ARGS --seed 456 --save_dir models/ensemble_3 >> ensemble_3.log 2>&1
echo "=== MODEL 3 DONE $(date) ===" >> ensemble_3.log

echo "=== ALL ENSEMBLE MODELS DONE $(date) ===" > ensemble_done.txt
