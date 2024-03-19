FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 as cuda-base
# Set some environment variable for better NVIDIA compatibility
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory to /app
WORKDIR /app
# Copy the inference folder to /app/inference
COPY /inference /app/inference

# Install curl and add the NodeSource repositories
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.9
RUN apt-get update && apt-get install -y curl wget ffmpeg unzip git python3-pip
# Update pip and install requirements
RUN pip install --upgrade pip
RUN pip install torch==1.13.1+cu117 torchvision>=0.13.1+cu117 torchaudio>=0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --no-cache-dir 
RUN pip install  -r inference/requirements.txt
RUN pip install uvicorn

# Download AzCopy
RUN wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux && tar -xf azcopy.tar.gz --strip-components=1

# Move AzCopy to the /usr/bin directory
RUN mv azcopy /usr/bin/

WORKDIR /app/inference
# Download the cloning model
RUN wget https://myshell-public-repo-hosting.s3.amazonaws.com/checkpoints_1226.zip && \
    unzip -o checkpoints_1226.zip && \
    rm -r checkpoints_1226.zip

CMD ["uvicorn", "fast:app", "--host", "::", "--port", "80"]
