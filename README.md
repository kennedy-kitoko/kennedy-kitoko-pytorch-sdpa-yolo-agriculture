# kennedy-kitoko-pytorch-sdpa-yolo-agriculture

# 🚀 SDPA-YOLO12: PyTorch SDPA as Flash Attention Alternative

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/abs/YOUR_PAPER)

> **🇨🇩 Innovation by Kennedy Kitoko** - Congolese AI Student  
> Simplifying AI Agriculture through Native PyTorch SDPA

## 🎯 **Key Innovation**

This repository presents **PyTorch SDPA** as a simplified, high-performance alternative to Flash Attention for YOLO12 models, specifically optimized for agricultural AI applications.

### 🏆 **Results Summary**
- **mAP@50**: 97.8% (vs 98.2% Flash Attention)
- **Setup Time**: 0 minutes (vs 30-60 min Flash Attention)
- **Dependencies**: 0 external (vs multiple complex for FA)
- **Compatibility**: Universal PyTorch 2.0+ (vs CUDA-specific)

## 🚀 **Quick Start**

```bash
# Clone repository
git clone https://github.com/KennedyKitoko/SDPA-YOLO12-SmartFarm
cd SDPA-YOLO12-SmartFarm

# Install dependencies (30 seconds)
pip install -r requirements.txt

# Run demo inference
python scripts/demo_inference.py --image data/sample_images/weed_sample_1.jpg

# Train your model
python src/train_sdpa.py --config configs/yolo12n_sdpa.yaml
