FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib64:$LD_LIBRARY_PATH
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# System update and add deadsnakes PPA for Python 3.12
RUN apt update -y && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt update -y && \
    apt install -y wget gnupg

# Add CUDA repository keyring
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb

# System dependencies installation
RUN apt-get update && apt-get install -y \
    wget unzip git build-essential cmake ninja-build \
    python3.11 python3.11-dev python3.11-venv python3-pip python3-wheel \
    libjpeg-dev libpng-dev \
    libgl1 libegl1 libgles2 libglvnd0 libosmesa6 \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libxrender1 \
    libgomp1 bash \
    cuda-toolkit-12-8 cuda-nvcc-12-8 cuda-libraries-dev-12-8 cuda-nvtx-12-8 \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11 setup
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 120 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 120

# Python packages - General
COPY docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install --ignore-installed -r /tmp/requirements.txt

# Install Kaolin
RUN pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

# Flash Attention 2.8.3 
RUN python -m pip install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

# BEN2 
RUN python -m pip install -e "git+https://github.com/PramaLLC/BEN2.git@2c99a5da477b5523585bfa5c893888a6e818a8f6#egg=ben2"

# Install CuMesh
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh && \
    cd /tmp/CuMesh && \
    git checkout c4ad6125924fcedfd13f0bd61520ca2d24eb7a87 && \
    git submodule update --init --recursive && \
    pip install /tmp/CuMesh --no-build-isolation && \
    rm -rf /tmp/CuMesh

# Install FlexGEMM
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM && \
    cd /tmp/FlexGEMM && \
    git checkout 6a184e957e1d3d18763221513b3e810ac1e7c8e8 && \
    git submodule update --init --recursive && \
    pip install /tmp/FlexGEMM --no-build-isolation && \
    rm -rf /tmp/FlexGEMM
    
# O-Voxel library 
COPY pipeline_service/libs/o-voxel /tmp/o-voxel
RUN git clone https://gitlab.com/libeigen/eigen.git /tmp/o-voxel/third_party/eigen && \
    cd /tmp/o-voxel/third_party/eigen && \
    git checkout 3147391d946bb4b6c68edd901f2add6ac1f31f8c && \
    pip install /tmp/o-voxel --no-build-isolation && \
    rm -rf /tmp/o-voxel

# App workspace
WORKDIR /workspace

# Application files
COPY pipeline_service /workspace
COPY configuration.yaml /workspace/configuration.yaml

EXPOSE 10006
CMD ["python", "serve.py"]
