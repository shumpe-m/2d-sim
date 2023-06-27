FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install pkg
RUN apt-get update && apt-get install -y \
	sudo \
    vim \
    wget \
    unzip \
    git \
    iputils-ping \
	net-tools \
    curl \
	init \
    build-essential \
    gcc \
    g++ \
    gnupg \
    make \
    libtool \
    texinfo \
    dpkg-dev \
    pkg-config \
    libgl1-mesa-glx \
    libcurl4-openssl-dev \
    libssl-dev \
    python3 \
    python3-dev \
    python3-setuptools \
    python3-tk \
    python3-pip

RUN pip3 install \
    torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html 
RUN pip3 install \
    numpy \
    opencv-python \
    matplotlib \
    tensorboard \
    scikit-learn \
    loguru

RUN echo "cd /root/2D-sim/scripts" >> ~/.bashrc
ENV PYTHONPATH="/root/2D-sim/scripts:$PYTHONPATH"





