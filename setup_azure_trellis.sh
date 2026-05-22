#!/usr/bin/env bash
# ==============================================================================
# Microsoft Research TRELLIS Installation & CAD Pipeline Setup Script for Azure
# ==============================================================================
# Designed specifically for Azure Data Science Virtual Machine (Ubuntu 22.04)
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

echo "======================================================================"
echo "🏗️  Starting Automated TRELLIS & Architectural CAD Pipeline Setup"
echo "======================================================================"

# 1. Update system packages
echo "[*] Updating system package definitions..."
sudo apt-get update -y
sudo apt-get install -y git build-essential ninja-build python3-dev

# 2. Check for CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo "[!] Warning: nvcc (CUDA Compiler) was not found in PATH."
    echo "[*] Attempting to find CUDA in standard Azure DSVM locations..."
    if [ -d "/usr/local/cuda" ]; then
        export PATH=/usr/local/cuda/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        echo "[+] CUDA path set to /usr/local/cuda"
    else
        echo "[Error] CUDA compiler (nvcc) is required to build TRELLIS extensions."
        echo "        Please make sure you are using an Azure GPU-enabled VM size."
        exit 1
    fi
fi
nvcc --version

# 3. Setup Conda environment
echo "[*] Configuring dedicated Conda environment 'trellis'..."
# Source conda commands
CONDA_PATH=$(conda info --base 2>/dev/null || echo "/anaconda")
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PATH/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

if conda info --envs | grep -q "trellis"; then
    echo "[+] Conda environment 'trellis' already exists. Activating..."
    conda activate trellis
else
    echo "[*] Creating Conda environment 'trellis' with Python 3.10..."
    conda create -n trellis python=3.10 -y
    conda activate trellis
fi

# 4. Install CUDA-compatible PyTorch
echo "[*] Installing PyTorch with CUDA 12.1 runtime..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Clone and setup Microsoft TRELLIS
if [ -d "TRELLIS" ]; then
    echo "[+] TRELLIS repository already cloned. Updating..."
    cd TRELLIS
    git pull
    cd ..
else
    echo "[*] Cloning official Microsoft TRELLIS repository..."
    git clone --recursive https://github.com/Microsoft/TRELLIS.git
fi

# 6. Install pip dependencies
echo "[*] Installing standard Python packages for TRELLIS..."
cd TRELLIS
pip install -r requirements.txt
pip install trimesh trimesh[easy] fast_simplification opencv-python Pillow -q
cd ..

# 7. Compile high-performance CUDA extensions
echo "[*] Beginning parallel compilation of specialized CUDA backends..."
echo "    (This might take 3-5 minutes, please do not close the terminal)"

export FORCE_CUDA=1
export TCNN_CUDA_ARCHITECTURES=86  # Standard for NVIDIA A10G / A10 GPUs

# A. Install nvdiffrast (NVIDIA high-performance modular rasterizer)
echo "[*] Compiling nvdiffrast..."
pip install git+https://github.com/NVlabs/nvdiffrast.git

# B. Install spconv (Sparse Convolution Library)
echo "[*] Compiling spconv..."
pip install git+https://github.com/EasternSun5566/spconv.git

# C. Install diff-gaussian-rasterization (for 3D Gaussian Splatting)
echo "[*] Compiling diff-gaussian-rasterization..."
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git

# 8. Verification
echo "======================================================================"
echo "🎉 Setup Completed Successfully!"
echo "======================================================================"
echo "To activate this environment in the future, run:"
echo "    conda activate trellis"
echo ""
echo "To test the pipeline, run:"
echo "    python trellis_cad_pipeline.py --image <your_sketch_image.png>"
echo "======================================================================"
