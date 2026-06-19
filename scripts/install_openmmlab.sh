#!/bin/bash
set -e

pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim
mim install "mmcv>=2.0.0rc4, <2.2.0"
mim install mmdet
mim install mmpose
pip install "numpy<2.0"