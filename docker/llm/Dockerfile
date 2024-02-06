FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
LABEL description="Llama-CPP-Python server (docs: https://github.com/abetlen/llama-cpp-python)."
# Install required tools
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
RUN python3 -m pip install --no-cache-dir --upgrade pip
# Install depencencies
RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context
# Install llama-cpp-python (build with cuda)
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
# Create working directory
RUN mkdir /app
RUN mkdir /app/llm_cstk
# Set current working directory
WORKDIR /app/llm_cstk
# Set host to allow external access
ENV HOST 0.0.0.0
# Port mappig
EXPOSE 8000