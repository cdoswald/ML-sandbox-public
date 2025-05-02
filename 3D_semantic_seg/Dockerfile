# Base image
FROM continuumio/miniconda3

# Install Linux packages
RUN apt-get update && apt-get install -y \
	git \
	libegl1 \
	libgl1 \
	libgomp1 \
	xvfb \
	&& rm -rf /var/lib/apt/lists/*

# Copy environment file
COPY environment.yml /workspace/environment.yml

# Initialize Conda in shell and activate base
SHELL ["/bin/bash", "-c"]
RUN conda init bash && \
	echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
	echo "conda activate base" >> ~/.bashrc

# Create Conda environment
RUN conda env create -f /workspace/environment.yml --platform linux-64

# Clone Waymo repo
RUN source ~/.bashrc && \
	conda activate waymo_open_env && \
	git clone https://github.com/waymo-research/waymo-open-dataset.git /workspace/waymo-od && \
	pip install waymo-open-dataset-tf-2-12-0==1.6.4

# Set default env
ENV CONDA_DEFAULT_ENV=waymo_open_env
ENV PATH /opt/conda/envs/waymo_open_env/bin:$PATH