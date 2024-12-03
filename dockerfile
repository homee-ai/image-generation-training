# Use the official Ubuntu 20.04 as a base image
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Prevent interactive prompts during package installations
ARG DEBIAN_FRONTEND=noninteractive

# Update the package list and install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install required Python packages    
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    Cython \
    numpy \
    ffmpeg \
    opencv-python-headless \
    pillow \
    accelerate \
    xformers \
    transformers \
    controlnet-aux \
    mediapipe \
    diffusers \
    prodigyopt \
    pytest 

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

CMD ["python3", "-u", "worker.py"]