#!/bin/bash
set -e

# mim install mmengine==0.10.3
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim "numpy<2.0"
mim install mmcv
mim install mmdet
mim install mmpose