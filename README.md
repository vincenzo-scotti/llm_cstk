# LLM CSTK

An *LLM*-powered toolkit for *chatbot* and *search* services.

This toolkit is thought to provide easy-to-deploy APIs integrating chatbot and (semantic) search services using LLM technologies.
The toolkit mainly provides two APIs, one for chatting and one for searching.
Additionally, we provide code for training *custom (large) language models* for chatting or *custom ranking models* for searching using *domain-specific data*, which can be easily integrated into the overall pipeline.

## Repository structure

This repository is organised into the following directories:

```
|- experiments/
  |- ...
|- notebooks/
  |- ...
|- resources/
  |- configs/
    |- ...
  |- data/
    |- ...
  |- models/
    |- ...
|- src/
  |- script/
    |- ...
  |- llm_cstk/
    |- ...
```

For further details, refer to the `README.md` within each directory.

## Setup

To install all the required packages within an [Anaconda](https://anaconda.org) environment, run the following commands:

```bash
# Create an Anaconda environment
conda create -n llm_cstk python=3.10 cudatoolkit=11.6
# Activate anaconda environment
conda activate llm_cstk
# Install packages
pip install requirements.txt
```

> [!NOTE]  
> Skip the `cudatoolkit` option if you don't want to use the GPU.

> [!WARNING]  
> This toolkit uses the [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python/tree/main) library; to use the GPU, follow the installation instructions on the library repository.

To add the source code directory to the Python path, you can add this line to the file `~/.bashrc`

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/llm_cstk/src
```

## Web API

In the following, we provide examples of how to use the Web API for chatting and searching.

### Chat

Most of the functionalities are powered using an LLM.
The code was initially developed to use the [Llama](https://arxiv.org/abs/2302.13971) and [Llama 2](https://arxiv.org/abs/2307.09288) models available through the [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python/tree/main) library.
In most cases, the service hosting the LLM is started transparently.

The chat-related functionalities are:
- **Information extraction**: given a reference document, the user interacts with an LLM-based chatbot to extract relevant information.
- **Knowledge-based question answering**: the user discusses with an LLM-based chatbot that can exploit external information from a knowledge base to answer.
- **Response generation**: The chatbot is used to suggest possible responses in a conversation.

#### Information extraction

```python
>>> import requests
>>> url = '...'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

#### Knowledge-based question answering

```python
>>> import requests
>>> url = '...'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

#### Response generation

...

##### Custom (L)LMs

You can use a language model fine-tuned on domain-specific data to generate candidate responses.

```python
>>> import requests
>>> url = '...'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

##### LLMs

LLMs can be used to:
- Elaborate on the responses suggested by the language models fine-tuned on domain-specific data to generate a suggested response.
- Exploit external knowledge (previous chats, as in few-shot learning or reference documents) to generate a suggested response.
- Use both previous approaches.

```python
>>> import requests
>>> url = '...'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

### Search

The search-related functionalities are:
- **Retrieve document (passage)**: find semantically or lexically similar documents (or document passages) to a given query.
- **Generating a snippet**: given the query results, highlight the passages more relevant to the query.
- **Adding new corpora**: add a new document collection to the search in.

#### Retrieve document (passage)

```python
>>> import requests
>>> url = '...'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

##### Document

```python
>>> import requests
>>> url = '...'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

##### Document passage

```python
>>> import requests
>>> url = '...'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

##### Document (long query)

```python
>>> import requests
>>> url = '...'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

##### Document passage (long query)

```python
>>> import requests
>>> url = '...'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

#### Generate snippet

```python
>>> import requests
>>> url = '...'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

##### Results

```python
>>> import requests
>>> url = '...'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

##### Results (long query)

```python
>>> import requests
>>> url = '...'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

#### Add new corpus

```python
>>> import requests
>>> url = '...'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

##### Insert in collection

```python
>>> import requests
>>> url = '...'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

##### Indexing

```python
>>> import requests
>>> url = '...'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

## Fine-tuning

We offer the possibility to fine-tune language models on domain-specific data for chatting or searching.

### Chat

There are scripts to fine-tune language models or large language models on domain-specific data.
The scripts expect to have the `./src` directory in the Python path and all data sets to be downloaded and placed in the `./resources/data/raw/` directory.

#### LM

In the following, we provide the instructions to fine-tune one of the language models available in the [Transformers library](https://huggingface.co/docs/transformers/index) from [Huggingface]().
Additionally, we provide instructions to monitor the training process.

##### Run

To fine-tune a language model on domain-specific data, run:

```bash
python ./src/bin/train_dialogue_nn.py --config_file_path ./resources/configs/path/to/training/config.yaml
```

To fine-tune the language model in background, run:

```bash
nohup python ./src/bin/train_dialogue_nn.py --config_file_path ./resources/configs/path/to/training/config.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out &
```

##### Monitor

It is possible to monitor the fine-tuning process using [Tensorboard](https://www.tensorflow.org/tensorboard).

To connect to a remote server and monitor the fine-tuning process, connect via ssh to your machine using a tunnel

```bash
ssh  -L 16006:127.0.0.1:6006 user@adderess
```

Start the Tensorboard server on the remote or local machine

```bash
tensorboard --logdir ./expertiments/path/to/tensorboard/
```

Finally, connect to http://127.0.0.1:6006 or http://127.0.0.1:16006 on your local machine, depending, respectively, whether the language model is fine-tuned on the local machine or a remote machine.

> [!NOTE]  
> Skip the ssh tunnel passage if you are locally connected to the machine you are using for fine-tuning.

#### LLM

...

### Search

For now, we suggest you train custom search models for ranking using the utilities from the [Sentence-Transformer library](https://www.sbert.net), which is the core of the search services.

## Deployiment

### Containers 

...

### Running instance

...

## Acknoledgements

- Vincenzo Scotti: ([vincenzo.scotti@polimi.it](mailto:vincenzo.scotti@polimi.it))
- Mark James Carman: ([mark.carman@polimi.it](mailto:mark.carman@.polimi.it))
