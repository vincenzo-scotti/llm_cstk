# LLM CSTK

An LLM-powered toolkit for chatbot and search services.

This toolkit is thought to provide easy-to-deploy APIs integrating chatbot and (semantic) search services using LLM technologies.
The toolkit mainly provides two APIs, one for chatting and one for searching.
Additionally, we provide code for training custom chatbots or sea on domain-specific data, which can be easily integrated into the overall pipeline.

## Setup

## Web API

### Chat

Most of the functionalities are powered using an LLM.
The code was initially developed to use the [Llama]() and [Llama 2]() models available through the []() library.
In most cases, the service hosting the LLM is started transparently.

#### Document analysis

#### Knowledge-based Q&A

...

#### Suggest responses



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
