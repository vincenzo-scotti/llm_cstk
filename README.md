# LLM CSTK

An *LLM*-powered toolkit for *chatbot* and *search* services.

This toolkit is thought to provide easy-to-deploy APIs integrating chatbot and (semantic) search services using LLM technologies.
The toolkit mainly provides two APIs, one for chatting and one for searching.
Additionally, we provide code for training *custom language models* for chatting or *custom ranking models* for searching using *domain-specific data*, which can be easily integrated into the overall pipeline.

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

### Chat

Most of the functionalities are powered using an LLM.
The code was initially developed to use the [Llama](https://arxiv.org/abs/2302.13971) and [Llama 2](https://arxiv.org/abs/2307.09288) models available through the [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python/tree/main) library.
In most cases, the service hosting the LLM is started transparently.

The chat-related functionalities are:
- *Document analysis*: 

#### Document analysis

...

#### Knowledge-based Q&A

...

#### Suggest responses

...

##### Using custom models

You can use a language model fine-tuned on domain-specific data to suggest candidate responses.

##### Using LLMs

LLMs can be used to:
- Elaborate on the responses suggested by the language models fine-tuned on domain-specific data to generate a suggested response
- Exploit external knowledge (previous chats, as in few-shot learning or reference documents) to generate a suggested response
- Use both previous approaches.

### Search

...

## Training

We offer the possibility to train language models on domain-specific data for chatting or searching.

### Chat

#### Fine-tuning LM

...

#### Fine-tuning LLM

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
