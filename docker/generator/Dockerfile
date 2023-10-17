FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
LABEL description="LM-based REST API for generator functionalities (docs: https://github.com/vincenzo-scotti/llm_cstk)."
# Install required tools
RUN apt update
RUN apt install -y build-essential git python3 python3-pip
RUN python3 -m pip install --no-cache-dir --upgrade pip
# Copy toolkit
COPY ../../llm_cstk .
# Set current working directory
WORKDIR ./llm_cstk
# Install required packages
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -r requirements.txt
# Update Python path
ENV PYTHONPATH "${PYTHONPATH}:./src"
# Set host to allow external access
ENV HOST 0.0.0.0
# Port mappig
EXPOSE 5000