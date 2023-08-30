FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL description="LM-based REST API for generator and retrieval functionalities (docs: https://github.com/vincenzo-scotti/llm_cstk)."

RUN apt update
RUN apt install -y build-essential git python3 python3-pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN git clone https://github.com/vincenzo-scotti/llm_cstk 
RUN cd llm_cstk

WORKDIR /llm_cstk

RUN python3 -m pip install -r requirements.txt

# CMD ["python3" "./src/script/retrieval_server.py" "--config_file_path" "./resources/configs/path/to/retrieval/config.yaml"]
