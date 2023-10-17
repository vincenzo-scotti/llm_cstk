FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
LABEL description="Llama-CPP-Python server (docs: https://github.com/abetlen/llama-cpp-python)."
# Install required tools
RUN apt update
RUN apt install -y build-essential git python3 python3-pip
RUN python3 -m pip install --no-cache-dir --upgrade pip
# Install required packages
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -r requirements.txt
# Set host to allow external access
ENV HOST 0.0.0.0
# Port mappig
EXPOSE 8000