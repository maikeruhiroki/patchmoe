# PatchMoE Development Environment
# Base image with CUDA support
FROM nvidia/cuda:12.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch==2.0.1+cu121 torchvision==0.15.2+cu121 torchaudio==2.0.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Python dependencies
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt

# Install additional ML libraries
RUN pip install \
    opencv-python \
    pillow \
    scikit-image \
    scikit-learn \
    seaborn \
    plotly \
    tensorboard \
    tqdm \
    optuna \
    kaggle

# Install Tutel (MoE library)
RUN pip install ninja
COPY tutel/ /workspace/tutel/
WORKDIR /workspace/tutel
RUN pip install -e .
WORKDIR /workspace

# Copy project files
COPY PatchMoE/ /workspace/PatchMoE/
COPY *.py /workspace/
COPY *.md /workspace/
COPY *.json /workspace/

# Create necessary directories
RUN mkdir -p /workspace/outputs \
    /workspace/medical_datasets \
    /workspace/real_medical_datasets \
    /workspace/real_medical_datasets_kaggle \
    /workspace/test_data

# Set Python path
ENV PYTHONPATH=/workspace

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Default command
CMD ["/bin/bash"]
