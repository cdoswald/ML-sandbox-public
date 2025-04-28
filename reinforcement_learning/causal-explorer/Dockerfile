FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# install Ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
    apt-get -y install \
        build-essential \
        cmake \
        curl \
        ffmpeg \
        git \
        libpython3-dev \
        libomp-dev \
        libopenblas-dev \
        libblas-dev \
        python3-dev \
        python3-opengl \
        python3-pip \
        wget \
        xvfb && \
	apt-get clean && rm -rf /var/lib/apt/lists/*
	
# creates symoblic link from python to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# install python dependencies for CleanRL and MuJoCo
RUN git clone https://github.com/vwxyzjn/cleanrl.git && \
	pip install -r cleanrl/requirements/requirements-mujoco.txt
	
# install additional python dependencies
RUN pip install \
	ffmpeg \
	imageio-ffmpeg \
	typing_extensions \
	gymnasium==0.29.1 \
	stable-baselines3==2.4.0 \
	torch --extra-index-url https://download.pytorch.org/whl/cu118
