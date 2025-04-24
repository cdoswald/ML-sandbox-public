FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    pkg-config \
    sudo \
    wget \
    unzip \
    ffmpeg \
    libopencv-dev \
    python3-opencv \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenjp2-7-dev \
    && apt-get clean

RUN pip install \
    opencv-python \
    pillow \
    torch \
    torchvision

RUN pip install \
    ultralytics \
	matplotlib \
    ipykernel \
    jupyter

WORKDIR /workspace