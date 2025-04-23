FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gdb \
    pkg-config \
    sudo \
    wget \
    unzip \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libeigen3-dev \
    libopencv-dev \
    libprotobuf-dev \
    protobuf-compiler \
    && apt-get clean

# Set environment variables
ENV ORT_VERSION=1.20.1
ENV ORT_DIR=/opt/onnxruntime

# Download and extract ONNX Runtime C++ binaries
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz && \
    tar -xzf onnxruntime-linux-x64-${ORT_VERSION}.tgz && \
    mv onnxruntime-linux-x64-${ORT_VERSION} ${ORT_DIR} && \
    rm onnxruntime-linux-x64-${ORT_VERSION}.tgz

# Set library path
ENV LD_LIBRARY_PATH="${ORT_DIR}/lib:${LD_LIBRARY_PATH}"
ENV CPLUS_INCLUDE_PATH="${ORT_DIR}/include"

# Set working directory
WORKDIR /workspace