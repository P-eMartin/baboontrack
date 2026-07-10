'''
This script checks if the required libraries are installed and can be imported successfully.
'''
import platform
import subprocess
# Check if the operating system is Windows
if platform.system() == "Windows":
    windows = True
else:
    windows = False
activate_env = ".\.venv\Scripts\\activate" if windows else "source .venv/bin/activate"
install_pip = "pip install .\dist\\baboontrack-0.0.1-py3-none-any.whl --force-reinstall" if windows else "pip install dist/baboontrack-0.0.1-py3-none-any.whl --force-reinstall"
install_openmmlab = "bash scripts\install_openmmlab.sh" if windows else "bash scripts/install_openmmlab.sh"

try:
    import torch
    print("PyTorch is installed. Version:", torch.__version__)
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
        # Check if GPU memory is available
        gpu_id = torch.cuda.current_device()
        gpu_tot = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        gpu_alloc = torch.cuda.memory_allocated(gpu_id) / 1024**3
        gpu_res = torch.cuda.memory_reserved(gpu_id) / 1024**3
        print(f"GPU memory (used/res/total): {gpu_alloc:.2f}/{gpu_res:.2f}/{gpu_tot:.2f} Gb")
    else:
        print("Warning: CUDA is not available. You may not be able to use GPU acceleration.")
except ImportError:
    print("Error: PyTorch is not installed. Please install it to proceed:")
    print(f"1. Activate your virtual environment: {activate_env}")
    print(f"2. Install the package: {install_pip}")
try:
    import mmcv
    print("mmcv is installed. Version:", mmcv.__version__)
    import mmdet
    print("mmdet is installed. Version:", mmdet.__version__)
    import mmpose
    print("mmpose is installed. Version:", mmpose.__version__)
except ImportError:
    print("Warning: mmcv is not installed. You may not be able to use PrimateFaceDetector which is facultative.")
    print("In order to use PrimateFaceDetector, you need to install the OpenMMLab libraries. Please follow these steps:")
    print(f"1. Activate your virtual environment: {activate_env}")
    print(f"2. Install the OpenMMLab libraries: {install_openmmlab}")

# Chech if conda environment sam3 exists
sam3_repo = "https://github.com/facebookresearch/sam3"
conda_env_name = "sam3"
conda_env_list = subprocess.run(["conda", "env", "list"], capture_output=True, text=True).stdout
if conda_env_name not in conda_env_list:
    print(f"Warning: Conda environment '{conda_env_name}' does not exist. You may not be able to use the SAM3-based detection and tracking, which is facultative.")
    print(f"Please create it by following the instructions in the SAM3 repository: {sam3_repo}")
else:
    print(f"Conda environment '{conda_env_name}' exists. If well installed, you may use the SAM3-based detection and tracking.")

# Check if baboontrack package is installed
try:
    import baboontrack
    from importlib.metadata import version
    print("baboontrack package is installed. Version:", version("baboontrack"))
except ImportError:
    print("Error: baboontrack package is not installed. Please install it to proceed:")
    print(f"1. Activate your virtual environment: {activate_env}")
    print(f"2. Install the package: {install_pip}")

