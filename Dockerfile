# Use the official NVIDIA CUDA devel image as a base
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# Set the environment variable for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.8
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.8 get-pip.py && rm get-pip.py

# Upgrade pip
RUN pip3.8 install --upgrade pip

# Install PyTorch, TorchVision, and Torchaudio with the specified versions and CUDA support
RUN pip3.8 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install additional Python packages
RUN pip3.8 install tqdm plyfile timm open_clip_torch scipy six configargparse pysocks python-dateutil imageio seaborn opencv-python scikit-learn tensorboard Pillow==9.5.0

# Verify installation
RUN python3.8 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Set python3.8 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Set environment variables for Python path
ENV PYTHONPATH=${PYTHONPATH}:/workspace/LEGaussians/submodules/diff-gaussian-rasterization
ENV PYTHONPATH=${PYTHONPATH}:/workspace/LEGaussians/submodules/simple-knn

# Set the working directory
WORKDIR /workspace/LEGaussians

# Entry point to keep the container running
CMD ["bash"]

