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
|- docker/
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
> This toolkit uses the [`llama-cpp-python` library](https://github.com/abetlen/llama-cpp-python/tree/main); to use the GPU, follow the installation instructions on the library repository.

To add the source code directory to the Python path, you can add this line to the file `~/.bashrc`

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/llm_cstk/src
```

## Web API

In the following, we provide examples of how to use the Web APIs for chatting and searching.

In the examples, we use the [`requests` library](https://requests.readthedocs.io) for Python, which is part of the requirements.
We also assume that the services are available at the following URL: `http://127.0.0.1`.

### Chat

Most of the functionalities are powered using an LLM.
The code was initially developed to use the [Llama](https://arxiv.org/abs/2302.13971) and [Llama 2](https://arxiv.org/abs/2307.09288) models available through the [`llama-cpp-python` library](https://github.com/abetlen/llama-cpp-python/tree/main).
In most cases, the service hosting the LLM is started transparently.

The chat-related functionalities are:
- **Response generation**: The chatbot is used to suggest possible responses in a conversation.
- **Information extraction**: given a reference document, the user interacts with an LLM-based chatbot to extract relevant information.
- **Knowledge-based question answering**: the user discusses with an LLM-based chatbot that can exploit external information from a knowledge base to answer.

In the examples, we assume that the service is listening to port `8999`.

The most straightforward use of the chat API is to generate a response using either a LLM or a custom LM.
We provide a generic function encapsulating all chat functionalities and some specific functions to simplify the use of the chat system.

```python
>>> import requests
>>> url = 'http://127.0.0.1:8999/generate'
>>> req_data = {
...   
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
>>> url = 'http://127.0.0.1:8999/generate/candidate_responses/custom_lm'
>>> req_data = {
...   'params': {
...     'n_samples': 3,  # `type: int`, the number of samples to generate
...     'speaker': 'AI',  # `type: str`, the identifier of the response's speaker
...     'custom_prompt': 'The following is a conversation between an AI assistant and a human user. '  # `type: Optional[str]`, the instructions for the language model (optional)
...                      'The assistant provides the user with technical support about computer programming.',  
...     'info': 'Python application crashing at startup',  # `type: Optional[str]`, a short description of the ongoing dialogue (optional)
...     'utterances': [  # `type: List[Dict[str, str]]`, a list of dialogue utterances up to now, each utterance is a dictionary with speaker identifier and generated text
...       {'speaker': 'AI', 'text': 'Hello, how may I assist you?'},
...       {'speaker': 'User', 'text': 'I am trying to start a Python application, but every time it says there are missing packages and crashes.'}
...     ]
...   }
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  'candidate_reponses': [  # `type: List[Dict[str, str]]`, a list with candidate responses, each response is a dictionary with speaker identifier and generated text
    {'speaker': 'AI', 'text': 'Have you tried turning the computer off and on again?'},
    {'speaker': 'AI', 'text': 'Have you tried rebooting the system?'},
    {'speaker': 'AI', 'text': 'Are you using the correct Python environment?'}
  ]
}
```

##### LLMs

LLMs can be used to:
- Elaborate on the responses suggested by the language models fine-tuned on domain-specific data to generate a suggested response.
- Exploit external knowledge (previous chats, as in few-shot learning or reference documents) to generate a suggested response.
- Use both previous approaches.

```python
>>> import requests
>>> url = 'http://127.0.0.1:8999/generate/candidate_responses/llm'
>>> req_data = {
...   'params': {
...     'n_samples': 3,  # `type: int`, the number of samples to generate
...     'speaker': 'AI',  # `type: str`, the identifier of the response's speaker
...     'custom_prompt': 'The following is a conversation between an AI assistant and a human user. '  # `type: Optional[str]`, the instructions for the language model (optional)
...                      'The assistant provides the user with technical support about computer programming. '
...                      'When available, the assistant may use additional information from suggested responses and possibly relevant documents, elaborating the additional information if useful.',  
...     'info': 'Python application crashing at startup',  # `type: Optional[str]`, a short description of the ongoing dialogue (optional)
...     'utterances': [  # `type: List[Dict[str, str]]`, a list of dialogue utterances up to now, each utterance is a dictionary with speaker identifier and generated text
...       {'speaker': 'AI', 'text': 'Hello, how may I assist you?'},
...       {'speaker': 'User', 'text': 'I am trying to start a Python application, but every time it says there are missing packages and crashes.'}
...     ]
...     'candidate_reponses': [  # `type: Optional[List[Dict[str, str]]]`, a list with candidate responses, each response is a dictionary with speaker identifier and generated text, to help respond (optional)
...       {'speaker': 'AI', 'text': 'Have you tried turning the computer off and on again?'},
...       {'speaker': 'AI', 'text': 'Have you tried rebooting the system?'},
...       {'speaker': 'AI', 'text': 'Are you using the correct Python environment?'}
...     ]
...     'relevant_documents': [  # `type: Optional[List[str]]`, a list with possibly useful document (passages), each document is a string, to help respond (optional)
...       {'The definitive guide to rebooting a system suggests to (...)'},
...       {'Python application common startup errors include (...)'},
...       {'Restarting your computer for dummies: (...)'}
...     ]
...   }
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  'response': {'speaker': 'AI', 'text': 'I think you should reboot your system.'}
}
```

#### Information extraction

```python
>>> import requests
>>> url = 'http://127.0.0.1:8999/generate/info_extraction'
>>> req_data = {
...   'params': {
...     'custom_prompt': 'The following is a conversation between an AI assistant and a human user. '  # `type: Optional[str]`, the instructions for the language model (optional)
...                      'The assistant helps the user analyse a document and extract useful information from it',  
...     'document': 'GPU installation guide ...',  # `type: str`, the text of the document to analyse
...     'utterances': [  # `type: List[Dict[str, str]]`, a list of dialogue utterances up to now, each utterance is a dictionary with speaker identifier and generated text
...       {'speaker': 'AI', 'text': 'Hello, how may I assist you?'},
...       {'speaker': 'User', 'text': 'What is the document about?'}
...     ]
...   }
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  'response': {'speaker': 'AI', 'text': 'The document contains instructions to install a GPU on a computer.'}
}
```

#### Knowledge-based question answering

```python
>>> import requests
>>> url = 'http://127.0.0.1:8999/generate/kb_qa'
>>> req_data = {
...   'params': {
...     'n_samples': 3,  # `type: int`, the number of samples to generate
...     'speaker': 'AI',  # `type: str`, the identifier of the response's speaker
...     'custom_prompt': 'The following is a conversation between an AI assistant and a human user. '  # `type: Optional[str]`, the instructions for the language model (optional)
...                      'The assistant provides the user with technical support about computer programming. '
...                      'When available, the assistant may use additional information from suggested responses and possibly relevant documents, elaborating the additional information if useful.',  
...     'relevant_documents': [  # `type: List[str]`, a list with possibly useful document (passages), each document is a string, to help respond
...       '',
...       '',
...       ''
...     ]
...     'utterances': [  # `type: List[Dict[str, str]]`, a list of dialogue utterances up to now, each utterance is a dictionary with speaker identifier and generated text
...       {'speaker': 'AI', 'text': 'Hello, how may I assist you?'},
...       {'speaker': 'User', 'text': 'Who is stronger? Ken or Son Goku?'}
...     ]
...   }
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  'response': {'speaker': 'AI', 'text': 'Son Goku, of course.'}
}
```

### Search

The search-related functionalities are:
- **Retrieve document (passage)**: find semantically or lexically similar documents (or document passages) to a given query.
- **Generating a snippet**: given the query results, highlight the passages more relevant to the query.
- **Adding new corpora**: add a new document collection to search over.

In the examples, we assume that the service is listening to port `8666`.

#### Retrieve document (passage)

The most straightforward use of the search API is to retrieve a document or a document passage from a collection.
We provide a generic function encapsulating all retrieval functionalities and some specific functions to simplify the use of the retrieval system.

```python
>>> import requests
>>> url = 'http://127.0.0.1:8666/search'
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

To search a document in a given collection using a simple query, use the `search_doc` function.
This is useful for building a search engine.

> [!NOTE]  
> Search can be divided among document passages to obtain more precise results.

```python
>>> import requests
>>> url = 'http://127.0.0.1:8666/search/doc'
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

To search a document passage in a given collection using a simple query, use the `search_doc_chunk` function.
This is useful for the knowledge-based question-answering chat function, which usually requires only a portion of a document to answer.

```python
>>> import requests
>>> url = 'http://127.0.0.1:8666/search/doc_chunk'
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

To search a document in a given collection using a long query, use the `search_doc_long_query` function.
This is useful for searching documents similar to a reference one.

> [!NOTE]  
> The query can be divided manually in multiple chunks to obtain more precise results.

```python
>>> import requests
>>> url = 'http://127.0.0.1:8666/search/doc_long_query'
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

To search a document passage in a given collection using a long query, use the `search_doc_long_query` function.
This is useful for searching documents similar to a reference one and using the response suggestions chat function, which usually requires only a portion of a document to see examples of responses.

```python
>>> import requests
>>> url = 'http://127.0.0.1:8666/search/doc_chunk_long_query'
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

Snippets are query-biased summaries of documents that usually provide a preview of the relevant content in a document for a specific query. 
We provide a generic function encapsulating all snippet generation functionalities and some specific functions to simplify the use of the snippet generation system.

```python
>>> import requests
>>> url = 'http://127.0.0.1:8666/snippet'
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

To generate a snippet given the search results obtained from a simple query, use the `generate_snippet` function.
This is useful for providing a preview in a search engine.

```python
>>> import requests
>>> url = 'http://127.0.0.1:8666/snippet/generate'
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

To generate a snippet given the search results obtained from a long query, use the `generate_snippet_long_query` function.
This is useful for providing a preview when searching for documents similar to a reference one.

```python
>>> import requests
>>> url = 'http://127.0.0.1:8666/snippet/generate_long_query'
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
>>> url = 'http://127.0.0.1:8666/corpus'
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
>>> url = 'http://127.0.0.1:8666/corpus/add'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

##### Insert in collection (large corpus)

```python
>>> import requests
>>> url = 'http://127.0.0.1:8666/corpus/add_large'
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
>>> url = 'http://127.0.0.1:8666/corpus/index'
>>> req_data = {
...   ...
... }
>>> output = requests.post(url, data=req_data).json
>>> print(output)
{
  ...
}
```

##### Indexing (large corpus)

```python
>>> import requests
>>> url = 'http://127.0.0.1:8666/corpus/index_large'
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

In the following, we provide instructions on fine-tuning language models on domain-specific data for chatting or searching.

All fine-tuning scripts require configurations provided via YAML files; for further details, refer to the examples in the `./resources/configs/` directory.

### Chat

There are scripts to fine-tune language models or large language models on domain-specific data.
The scripts expect the `./src/` directory in the Python path.

#### LM

In the following, we provide the instructions to fine-tune one of the language models available in the [Transformers library](https://huggingface.co/docs/transformers/index) from [Huggingface]().
Additionally, we provide instructions to monitor the fine-tuning process.

##### Run

To fine-tune a language model on domain-specific data, run:

```bash
python ./src/script/fine_tune_lm.py --config_file_path ./resources/configs/path/to/training/config.yaml
```

To fine-tune the language model in background, run:

```bash
nohup python ./src/script/fine_tune_lm.py --config_file_path ./resources/configs/path/to/training/config.yaml > experiment_"$(date '+%Y_%m_%d_%H_%M_%S')".out &
```

> [!NOTE]  
> The fine-tuning script works with both *causal* (or *decoder-only*) language models and *transducer* (or *encoder-decoder*) language models.

##### Monitor

It is possible to monitor the fine-tuning process using [Tensorboard](https://www.tensorflow.org/tensorboard).

To connect to a remote server and monitor the fine-tuning process, connect via ssh to your machine using a tunnel

```bash
ssh user@adderess -L 16006:127.0.0.1:6006
```

Start the Tensorboard server on the remote or local machine

```bash
tensorboard --logdir ./expertiments/path/to/tensorboard/
```

Finally, connect to http://127.0.0.1:6006 or http://127.0.0.1:16006 on your local machine, depending, respectively, whether the language model is fine-tuned on the local machine or a remote machine.

> [!NOTE]  
> Skip the ssh tunnel passage if you are locally connected to the machine you use for fine-tuning.

#### LLM

...

### Search

For now, we suggest you train custom search models for ranking using the utilities from the [Sentence-Transformer library](https://www.sbert.net), which is the core of the search services.

## Deployiment

In the following, we provide instructions on how to deploy the Web APIs for chatting and searching.

All servers require configurations provided via YAML files; for further details, refer to the examples in the `./resources/configs/` directory.

### Containers 

... [Docker](https://www.docker.com) container ...

#### Chat

...

#### Search

...

### Running manually

The scripts expect the `./src/` directory in the Python path.

#### Chat

To start the chat service in foreground, run:

```bash
python ./src/script/generator_server.py --config_file_path ./resources/configs/path/to/generator/config.yaml
```

To start the chat service in background, run:

```bash
nohup python ./src/script/generator_server.py --config_file_path ./resources/configs/path/to/generator/config.yaml > generator_server_"$(date '+%Y_%m_%d_%H_%M_%S')".out &
```

#### Search

To start the retrieval (search) service in foreground, run:

```bash
python ./src/script/retrieval_server.py --config_file_path ./resources/configs/path/to/retrieval/config.yaml
```

To start the retrieval service in background, run:

```bash
nohup python ./src/script/retrieval_server.py --config_file_path ./resources/configs/path/to/retrieval/config.yaml > retrieval_server_"$(date '+%Y_%m_%d_%H_%M_%S')".out &
```

## Acknowledgements


- Vincenzo Scotti: ([vincenzo.scotti@polimi.it](mailto:vincenzo.scotti@polimi.it))
- Mark James Carman: ([mark.carman@polimi.it](mailto:mark.carman@.polimi.it))
