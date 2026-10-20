#!/usr/bin/env bash
# Reproduce every number and figure in the README, end to end.
# About 2.5 hours on an RTX 4060 Laptop GPU. Run from the repository root with the environment active:
#   bash scripts/reproduce.sh /path/to/raw/animals      # folder containing train/ and test/
set -euo pipefail
RAW_DATA=${1:?"usage: bash scripts/reproduce.sh RAW_DATA_DIR"}

# 1. Data: decode, fix transparency, deduplicate, stratified train/val split.
python scripts/prepare_data.py --src "$RAW_DATA" --out data/processed
python scripts/dedup_calibration.py   # evidence for the near-duplicate threshold

# 2. Training + one-off test evaluation. The teacher (convnext_tiny) must come before the student.
for exp in convnext_tiny efficientnet_b0 resnet50 simple_cnn_baseline simple_cnn \
    simple_cnn_class_weighted simple_cnn_distill; do
  python scripts/train.py --config "configs/$exp.yaml" --overwrite
  python scripts/evaluate.py --checkpoint "outputs/$exp/best.pt"
done

# 3. CLIP zero-shot and linear-probe baselines.
python scripts/clip_eval.py

# 4. Deployment: ONNX export, INT8 (setting chosen on val), test accuracy; then latency of all
#    models in one session (run it on an idle machine).
for exp in convnext_tiny efficientnet_b0 resnet50 simple_cnn simple_cnn_distill; do
  python scripts/deploy.py --checkpoint "outputs/$exp/best.pt"
done
python scripts/benchmark.py

# 5. Explainability and error analysis.
python scripts/gradcam_grid.py --checkpoints \
  outputs/simple_cnn/best.pt outputs/simple_cnn_distill/best.pt outputs/convnext_tiny/best.pt
python scripts/error_analysis.py --predictions outputs/convnext_tiny/test_predictions.npz

# 6. Tables.
python scripts/make_report.py
