# Base image
FROM mambaorg/micromamba:2.3.0

# Install Linux packages
USER root
RUN apt-get update && apt-get install -y \
	git \
	libegl1 \
	libgl1 \
	libgomp1 \
	xvfb \
	&& rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# Set environment activation variable
ENV MAMBA_DOCKERFILE_ACTIVATE=1

# Copy and install environment
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /workspace/environment.yml
RUN micromamba create -y -n waymo_open_env -f /workspace/environment.yml && \
	micromamba clean --all --yes

# Clone Waymo repo
RUN git clone https://github.com/waymo-research/waymo-open-dataset.git /workspace/waymo-od && \
	micromamba run -n waymo_open_env pip install waymo-open-dataset-tf-2-12-0==1.6.4