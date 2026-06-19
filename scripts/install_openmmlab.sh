#!/bin/bash
set -e

uv pip install -U openmim "numpy<2.0"
# mim install mmengine==0.10.3
mim install mmcv==2.1.0
uv pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121
mim install mmdet==3.2.0
mim install mmpose==1.3.2