#!/bin/bash
# Run this once, right after SSH-ing into the rented Vast.ai instance.
#
# Assumes a "PyTorch" template instance (CUDA + cuDNN already present).
# Does NOT touch anything on your local machine.
set -e

echo "================================"
echo "System check"
echo "================================"
nvidia-smi
python3 --version

echo "================================"
echo "Python deps"
echo "================================"
pip install --upgrade pip || echo "pip self-upgrade skipped (system-managed pip), continuing"
pip install --upgrade transformers accelerate bitsandbytes
pip install opencv-python-headless pillow numpy pandas spacy tqdm torchvision
pip install qwen-vl-utils[decord]
python3 -m spacy download en_core_web_sm

echo "================================"
echo "Verify SAM3 is importable"
echo "================================"
python3 -c "
from transformers import Sam3Model, Sam3Processor
print('Sam3Model OK:', Sam3Model)
"

echo "================================"
echo "Pre-download model weights (so the smoke test doesn't eat rented time on downloads)"
echo "================================"
python3 -c "
from huggingface_hub import snapshot_download
print('Downloading facebook/sam3 ...')
snapshot_download('facebook/sam3')
print('Downloading Qwen/Qwen3-VL-8B-Instruct (4-bit will be applied at load time, full weights still need to be fetched) ...')
snapshot_download('Qwen/Qwen3-VL-8B-Instruct')
print('Done.')
"

echo "================================"
echo "Setup complete. Next: upload the 8 videos + src/ (see UPLOAD_CHECKLIST.md),"
echo "then run verify_models.py as a smoke test BEFORE building the real pipeline."
echo "================================"
