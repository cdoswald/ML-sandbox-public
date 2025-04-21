FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    make \
    gdb \
    sudo \
    && apt-get clean

WORKDIR /workspace