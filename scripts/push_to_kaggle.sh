#!/usr/bin/env bash
# Publishes the notebook to Kaggle as a private kernel (GPU on, internet on, PlantVillage attached).
# Needs the Kaggle API token in ~/.kaggle/kaggle.json and `pip install kaggle`.
# Before the first push, replace YOUR_KAGGLE_USERNAME in kaggle/kernel-metadata.json with your Kaggle username.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
cp "$ROOT/notebook/tomato.ipynb" "$ROOT/kaggle/kernel-metadata.json" "$STAGE/"
kaggle kernels push -p "$STAGE"
rm -rf "$STAGE"
echo "Pushed. Open the kernel on Kaggle, check Settings (GPU, Internet on), then Run All."
