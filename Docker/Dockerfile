FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04

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

RUN pip3 install tensorflow==2.8.0 \
    tensorflow-gpu==2.8.0 \
    numpy \
    opencv-python \
    matplotlib

RUN echo "cd /root/2D-sim" >> ~/.bashrc