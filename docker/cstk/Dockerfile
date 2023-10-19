FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
LABEL description="LM-based REST API for generator functionalities (docs: https://github.com/vincenzo-scotti/llm_cstk)."
# Install required tools
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip default-jdk default-jre
RUN python3 -m pip install --no-cache-dir --upgrade pip
# Create working directory
RUN mkdir /app
RUN mkdir /app/llm_cstk
# Set current working directory
WORKDIR /app/llm_cstk
# Copy toolkit
COPY . .
# Install required packages
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -r requirements.txt
# Update Python path
ENV PYTHONPATH "${PYTHONPATH}:/app/llm_cstk/src"
# Set host to allow external access
ENV HOST 0.0.0.0
# Port mappig
EXPOSE 5000