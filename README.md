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
conda create -n llm_cstk python=3.10 cudatoolkit=11.8
# Activate anaconda environment
conda activate llm_cstk
# Install packages
pip install -r requirements.txt
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
We describe the APIs individually ([chat](#chat) and [search](#search)).

### Chat

Most of the functionalities are powered using an LLM.

The chat-related functionalities are:

- **Response suggestion**: The chatbot is used to suggest possible responses in a conversation (either using a LLM or a custom LM).
- **Information extraction**: given a reference document, the user interacts with an LLM-based chatbot to extract relevant information.

In the examples, we assume that the service is listening to port `5000` and that there is a mapping between port `15991` and `5000`.

The most straightforward use of the chat API is to generate a response using either a LLM or a custom LM.
We provide a generic function encapsulating all chat functionalities and some specific functions to simplify the use of the chat system.

URL: `http://127.0.0.1:15991/generate/`

Input:

```json
{
  "params": {
    "model": "llm",
    "sample": "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\nCustomer care on Twitter -- Date 2017-10-26 22:00:10+00:00\n\nBrand: Ask Spectrum\n\nAbstract:\nThe customer is complaining that he is facing internet outtage issue.\nThe agent asked the customer tosend an update to the modem to see if it responds and it will reboot if they do that so.\n\nCustomer 423147: Experiencing an internet outtage- zipcode 44060\n---\nAskSpectrum: Your modem appears to be online and providing an IP address to an external Linksys router. Has service returned? ^PS\n---\nCustomer 423147: No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection\n---\nAskSpectrum:",
    "task": "response_suggestion",
    "corpus": "tweet_summ",
    "custom_generate_params": {
       "temperature": 0.667
    },
    "approach": "plain_completion"
  }
}
```

Output:

```json
{
    "speaker": "assistant",
    "text": "Sorry to hear that. Can you please check if the issue is specific to your devices or a neighborhood wide outage? You can also restart your modem and router by unplugging them for 30 seconds before powering them back on. ^PS"
}
```

#### Response suggestion

To get one or more response suggestions, use the `response_suggestion` function.
This is useful to offer suggestions to the users.

##### Custom (L)LMs

You can use a language model fine-tuned on domain-specific data to generate candidate responses.

URL: `http://127.0.0.1:15991/generate/response_suggestion/custom_lm`

Input:

```json
{
  "params": {
    "utterances": [
      {
        "speaker": "Customer 423147",
        "sys": false,
        "text": "Experiencing an internet outtage- zipcode 44060"
      },
      {
        "speaker": "AskSpectrum",
        "sys": false,
        "text": "Your modem appears to be online and providing an IP address to an external Linksys router. Has service returned? ^PS"
      },
      {
        "speaker": "Customer 423147",
        "sys": false,
        "text": "No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection"
      }
    ],
    "speaker": "AskSpectrum",
    "corpus": "tweet_summ",
    "info": "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\nCustomer care on Twitter -- Date 2017-10-26 22:00:10+00:00\n\nBrand: Ask Spectrum"
  }
}
```

Output:

```json
{
  "candidates": [
    {
      "speaker": "assistant",
      "text": "..."
    }
  ]
}
```

##### LLMs

Given a dialogue and a speaker (the identifier of the person that should respond) LLMs can be used to:
- Elaborate on the responses suggested by the language models fine-tuned on domain-specific data to generate a suggested response.
- Exploit external knowledge (previous chats, as in few-shot learning or reference documents) to generate a suggested response.
- Use both previous approaches.

URL: `http://127.0.0.1:15991/generate/response_suggestion/llm`

Input:

```json
{
  "params": {
    "utterances": [
      {
        "speaker": "Customer 423147",
        "sys": false,
        "text": "Experiencing an internet outtage- zipcode 44060"
      },
      {
        "speaker": "AskSpectrum",
        "sys": false,
        "text": "Your modem appears to be online and providing an IP address to an external Linksys router. Has service returned? ^PS"
      },
      {
        "speaker": "Customer 423147",
        "sys": false,
        "text": "No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection"
      }
    ],
    "speaker": "AskSpectrum",
    "corpus": "tweet_summ",
    "info": "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\nCustomer care on Twitter -- Date 2017-10-26 22:00:10+00:00\n\nBrand: Ask Spectrum"
  }
}
```

Output:

```json
{
    "candidates": [
        {
            "speaker": "assistant",
            "text": "Sorry to hear that, @423147! I'd like to assist you further. Can you please try restarting your router by unplugging it from the power source for 30 seconds and then plugging it back in? ^PS"
        }
    ]
}
```

Moreover, LLMs can be used in *few-shot learning* mode, providing examples of other dialogues and corresponding responses.
This is helpful to guide the LLM on unseen dialogues and tasks

> [!NOTE]
> 
> You can chunk the current dialogue and use it as a long query to search for similar conversations to be used as examples in the few shot lerning.
> For further details refer to the [Document (long query) section](#document--long-query-).

URL: `http://127.0.0.1:15991/generate/response_suggestion/llm`

Input:

```json
{
  "params": {
    "utterances": [
      {
        "speaker": "Customer 423147",
        "sys": false,
        "text": "Experiencing an internet outtage- zipcode 44060"
      },
      {
        "speaker": "AskSpectrum",
        "sys": false,
        "text": "Your modem appears to be online and providing an IP address to an external Linksys router. Has service returned? ^PS"
      },
      {
        "speaker": "Customer 423147",
        "sys": false,
        "text": "No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection"
      }
    ],
    "speaker": "AskSpectrum",
    "corpus": "tweet_summ",
    "info": "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\nCustomer care on Twitter -- Date 2017-10-26 22:00:10+00:00\n\nBrand: Ask Spectrum",
    "examples": [
      {
        "data_set_id": "tweet_summ",
        "dialogue_id": "1eaf47c50e23d44b83d441155046cb55",
        "info": "Customer Care Chat 1eaf47c50e23d44b83d441155046cb55\n\nCustomer care on Twitter -- Date 2017-11-30 06:01:35+00:00\n\nBrand: Ask Spectrum", 
        "utterances": [
          {
            "speaker": "Customer 120960",
            "sys": false,
            "text": "Why isn't my router going online????????? Answer that for me. It makes no sense at this point. It already drops connection atleast once a day and now it's been offline for awhile now. Enlighten me."
          },
          {
            "speaker": "AskSpectrum",
            "sys": false,
            "text": "Good evening! I would really like to help with your internet services. When you can plz DM your phone number and address to chat? ^DT"
          },
          {
            "speaker": "Customer 120960",
            "sys": false,
            "text": "done."
          },
          {
            "speaker": "AskSpectrum",
            "sys": false,
            "text": "Thank you for your response. It looks as though there has been some intermittent connectivity with your modem for the last few days. This would require a service appointment to have the matter addressed. We ask that you please send your general availabi... https://t.co/9NuFVZ7J5Z"
          },
          {
            "speaker": "Customer 120960",
            "sys": false,
            "text": "done again"
          },
          {
            "speaker": "Customer 120960",
            "sys": false,
            "text": "Is anyone going to come?"
          },
          {
            "speaker": "AskSpectrum",
            "sys": false,
            "text": "We are awaiting the best call back number, so we can setup the appointment for you. ^JH"
          },
          {
            "speaker": "AskSpectrum",
            "sys": false,
            "text": "Please provide the best call ahead number via DM, thanks. ^JH"
          },
          {
            "speaker": "Customer 120960",
            "sys": false,
            "text": "done"
          },
          {
            "speaker": "AskSpectrum",
            "sys": false,
            "text": "Thanks, I'm pulling up the schedule now. ^JH"
          },
          {
            "speaker": "Customer 120960",
            "sys": false,
            "text": "Okay, let me know please"
          }
        ],
        "title": "Customer Care Chat 1eaf47c50e23d44b83d441155046cb55"
      },
      {
        "data_set_id": "tweet_summ",
        "dialogue_id": "b32f712ec96129b009d6305e04444263",
        "info": "Customer Care Chat b32f712ec96129b009d6305e04444263\n\nCustomer care on Twitter -- Date 2017-10-17 13:03:06+00:00\n\nBrand: Ask Spectrum",
        "utterances": [
          {
            "speaker": "Customer 502246",
            "sys": false,
            "text": "Got a message from  saying I was a valued customer and with an offer. Offer is bogus! Bait and switch at its worst!"
          },
          {
            "speaker": "AskSpectrum",
            "sys": false,
            "text": "I am sorry we had not been able to provide that for you. Had you received the offer via USPS or email? ^JR"
          },
          {
            "speaker": "Customer 502246",
            "sys": false,
            "text": "USPS"
          },
          {
            "speaker": "AskSpectrum",
            "sys": false,
            "text": "Hello Ed, I apologize for the unpleasant experience. Can you DM us this offer information so that we can fully investigate it? ^ JMM"
          },
          {
            "speaker": "Customer 502246",
            "sys": false,
            "text": "I tossed it in the garbage. Offer was $39.95 each for internet and tv. Hidden fees pushed total to more than $100. Rep was nasty/insulting"
          },
          {
            "speaker": "AskSpectrum",
            "sys": false,
            "text": "Do you happen to remember the name of the representative you spoke with?  If you do, please DM us the name so tha... https://t.co/t9NPnTlvbl"
          },
          {
            "speaker": "Customer 502246",
            "sys": false,
            "text": "I believe it was \"Will\" but I can check to be sure. I pay about $60 for internet. I would have added $20 to have tv too but not $40+!"
          },
          {
            "speaker": "Customer 502246",
            "sys": false,
            "text": "The offer plainly said $39.95 EACH.  Your rep immediately jumped to about $100. I will keep my antenna! So much for customer loyalty."
          },
          {
            "speaker": "AskSpectrum",
            "sys": false,
            "text": "Our sales team will be able to give you the most up to date discounts and packages available at the present time.... https://t.co/t9NPnTlvbl"
          },
          {
            "speaker": "Customer 502246",
            "sys": false,
            "text": "You should not advertise a rate then add for cable box, fees, taxes, etc for a $25+ increase. Be up front with it! Don't hide it. Deceptive!"
          }
        ],
        "title": "Customer Care Chat b32f712ec96129b009d6305e04444263"
      }
    ]
  }
}
```

Output:

```json
{
    "candidates": [
        {
            "speaker": "assistant",
            "text": "I'd like to assist further! Please DM your account number or the first/last name on the bill so we can look into this issue. ^PS"
        }
    ]
}
```

#### Information extraction

To have a response for a dialogue analysing a given document, use the `info_estraction` function.
This is useful to analyse an existing document and get some insights on its content.

URL: `http://127.0.0.1:15991/generate/info_extraction`

Input:

```json
{
  "params": {
    "document": "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\nCustomer care on Twitter -- Date 2017-10-26 22:00:10+00:00\n\nBrand: Ask Spectrum\n\nAbstract:\nThe customer is complaining that he is facing internet outtage issue.\nThe agent asked the customer tosend an update to the modem to see if it responds and it will reboot if they do that so.\n\n- Customer 423147: Experiencing an internet outtage- zipcode 44060\n\n- AskSpectrum: Your modem appears to be online and providing an IP address to an external Linksys router. Has service returned? ^PS\n\n- Customer 423147: No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection\n\n- AskSpectrum: Would it be OK if I send an update to the modem to see if it responds? It will reboot if I do so. ^PS\n\n- Customer 423147: Yes, that is ok.\n\n- AskSpectrum: The modem has reset and still shows the connection to the connection to the Linksys router. Are you able to get online now? ^PS\n\n- Customer 423147: No, here is error I'm getting on phone...(LandofOoo = router name) https://t.co/FNQYDNP6lM\n\n- AskSpectrum: Do you have the ability to connect a device via Ethernet, either to the router or directly to the modem?",
    "utterances": [
      {
        "speaker": "assistant",
        "text": "Hello, how may I assist you?"
      },
      {
        "speaker": "user",
        "text": "Write a short summary of the document, please."
      }
    ], 
    "corpus": "tweet_summ"
  }
}
```

Output:

```json
{
  "response": {
    "speaker": "assistant",
    "text": "As a helpful assistant, I'd be happy to summarize the document for you.\n\nThe document is a customer support chat between Ask Spectrum and a customer (Customer 423147) on Twitter. The customer is experiencing an internet outage issue in their area with zipcode 44060. The agent asks the customer to send an update to the modem to see if it responds, which will reboot the modem. After doing so, the modem shows a connection to the Linksys router, but the customer still cannot get online. The agent then asks if the customer can connect a device via Ethernet to either the router or modem.\n\nThat's a summary of the document!"
  }
}
```

#### Query extraction

To check if an input piece of a dialogue contains a question/information request or not, use the `query_extraction` function.
This is useful to build a retrieval augmented generation chatbot.

URL: `http://127.0.0.1:15991/generate/query_extraction`

Input:

```json
{
   "params": {
      "snippet": "",
      "corpus": "L"
   }
}
```

Output:

```json
{
   "response": {
      "speaker": "assistant",
      "text": ""
   }
}
```

#### Query recognition

To check if an input sentence is a question/request or a normal search engine query, use the `query_recognition` function.
This is useful to build a question answering pipeline.

URL: `http://127.0.0.1:15991/generate/query_recognition`

Input:

```json
{
   "params": {
      "query": "How can I check the speed of my internet wired device?",
      "corpus": "tweet_summ"
   }
}
```

Output:

```json
{
    "response": {
        "speaker": "assistant",
        "text": "Yes."
    }
}
```

> [!NOTE]
>
> For this functionality the output text is always either "Yes." or "No."

#### Relevant document selection

To check if an input document (or document passage) is useful to answer a question/request, use the `relevant_document_selection` function.
This is useful to build a question answering pipeline.

URL: `http://127.0.0.1:15991/generate/relevant_document_selection`

Input:

```json
{
   "params": {
      "question": "How can I check the speed of my internet wired device?",
      "document": "Customer Care Chat 0d08a099fa2b67d7e733ddaec28eed84\n\n(...)\n\n- Customer 177474: Thanks for the response! It's via wifi - tested in different areas of the house and speeds vary dramatically. Quickest I've got is 34 down\n\n- VerizonSupport: Are you able to test the speed on a wired device? \n^TDC\n\n- Customer 177474: Wired: 98.15 down / 71.56 up . Might be wifi broadcasting? Using WRT 3200 ACM router .\n\n(...)",
      "corpus": "tweet_summ"
   }
}
```

Output:

```json
{
    "response": {
        "speaker": "assistant",
        "text": "Yes."
    }
}
```

> [!NOTE]
>
> For this functionality the output text is always either "Yes." or "No."

#### Knowledge-Based Question Answering

To have a response to a question, use the `kb_qa` function.
This is useful to analyse an existing document and get some insights on its content.

URL: `http://127.0.0.1:15991/generate/kb_qa`

Input:

```json
{
   "params": {
      "question": "How can I check the speed of my internet wired device?",
      "reference_documents": [
         "Customer Care Chat 0d08a099fa2b67d7e733ddaec28eed84\n\n(...)\n\n- Customer 177474: Thanks for the response! It's via wifi - tested in different areas of the house and speeds vary dramatically. Quickest I've got is 34 down\n\n- VerizonSupport: Are you able to test the speed on a wired device? \n^TDC\n\n- Customer 177474: Wired: 98.15 down / 71.56 up . Might be wifi broadcasting? Using WRT 3200 ACM router .\n\n(...)",
         "Customer Care Chat 0d08a099fa2b67d7e733ddaec28eed84\n\n- Customer 177474: Hey , Check out my  result! https://t.co/QNI1xmeqC5 #speedtest - I pay for 100/100 . Help please?\n\n- VerizonSupport: That's not good.  Are testing from a wired or wireless device?  ^CAR\n\n- Customer 177474: Thanks for the response! It's via wifi - tested in different areas of the house and speeds vary dramatically. Quickest I've got is 34 down\n\n(...)"
      ],
      "corpus": "tweet_summ"
   }
}
```

Output:

```json
{
    "response": {
        "speaker": "assistant",
        "text": "According to Document 1, you can check the speed of your internet wired device by testing it directly connected to the router using a wired connection. The customer in this chat had a speed test result of:\n\n* 98.15 Mbps (downstream)\n* 71.56 Mbps (upstream)\n\nThis suggests that the wired connection is capable of delivering faster speeds than the wireless connection, which was only able to achieve 34 Mbps downstream."
    }
}
```

### Search

The search-related functionalities are:

- **Retrieve document (passage)**: find semantically or lexically similar documents (or document passages) to a given query.
- **Generate a snippet**: given the query results, highlight the passages more relevant to the query.

In the examples, we assume that the service is listening to port `5000` and that there is a mapping between port `15990` and `5000`.

#### Retrieve document (passage)

The most straightforward use of the search API is to retrieve a document or a document passage from a collection.
We provide a generic function encapsulating all retrieval functionalities and some specific functions to simplify the use of the retrieval system.

URL: `http://127.0.0.1:15990/search`

Input:

```json
{
  "params": {
    "query": "network wired device",
    "corpus": "tweet_summ",
    "ranking": "semantic",
    "reranking": "semantic",
    "chunk_doc": true,
    "doc_score_aggregation": "max"
  }
}
```

Output:

```json
{
  "query": "network wired device",
  "docno": [
    "69b9cc7286db785acfc78d7613382ad9",
    "8d31903f7989dfb707641538e65fcea3",
    "3912147a9e51918abfffc44d8730cc52",
    ...
  ],
  "score": [
    0.0020253651309758425,
    0.0018677901243790984,
    0.0012569052632898092,
    ...
  ]
}
```

##### Document

To search a document in a given collection using a simple query, use the `search_doc` function.
This is useful for building a search engine.

> [!NOTE]  
> Search can be divided among document passages to obtain more precise results.

URL: `http://127.0.0.1:15990/search/doc`

Input:

```json
{
  "params": {
    "query": "connectivity issue",
    "corpus": "tweet_summ",
    "ranking": "semantic",
    "chunk": true
  }
}
```

Output:

```json
{
  "query": "connectivity issue",
  "docno": [
    "86026c34fe8db0849a9070c0dafa58b1",
    "8d31903f7989dfb707641538e65fcea3",
    "61c958a7774587c526640bf4ae2642b3",
    ...
  ],
  "score": [
    0.6285992860794067,
    0.6274384260177612,
    0.6134823560714722,
    ...
  ]
}
```

##### Document passage

To search a document passage in a given collection using a simple query, use the `search_doc_chunk` function.
This is useful for the knowledge-based question-answering chat function, which usually requires only a portion of a document to answer.

URL: `http://127.0.0.1:15990/search/doc_chunk`

Input:

```json
{
  "params": {
    "query": "connectivity issue",
    "corpus": "tweet_summ",
    "ranking": "semantic"
  }
}
```

Output:

```json
{
  "query": "connectivity issue",
  "docno": [
    "86026c34fe8db0849a9070c0dafa58b1%p3",
    "8d31903f7989dfb707641538e65fcea3%p3",
    "86026c34fe8db0849a9070c0dafa58b1%p1",
    ...
  ],
  "score": [
    0.6285992860794067,
    0.6274384260177612,
    0.624739408493042,
    ...
  ]
}
```

##### Document (long query)

To search a document in a given collection using a long query, use the `search_doc_long_query` function.
This is useful for searching documents similar to a reference one.

> [!NOTE]  
> The query can be divided manually in multiple chunks to obtain more precise results.

URL: `http://127.0.0.1:15990/search/doc_long_query`

Input:

```json
{
  "params": {
    "query": [
      "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\nCustomer care on Twitter -- Date 2017-10-26 22:00:10+00:00\n\nBrand: Ask Spectrum\n\nAbstract:\nThe customer is complaining that he is facing internet outtage issue.\nThe agent asked the customer tosend an update to the modem to see if it responds and it will reboot if they do that so.",
      "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n- Customer 423147: Experiencing an internet outtage- zipcode 44060\n\n- AskSpectrum: Your modem appears to be online and providing an IP address to an external Linksys router. Has service returned? ^PS\n\n- Customer 423147: No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection\n\n(...)",
      "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection\n\n- AskSpectrum: Would it be OK if I send an update to the modem to see if it responds? It will reboot if I do so. ^PS\n\n- Customer 423147: Yes, that is ok.\n\n(...)",
      "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: Yes, that is ok.\n\n- AskSpectrum: The modem has reset and still shows the connection to the connection to the Linksys router. Are you able to get online now? ^PS\n\n- Customer 423147: No, here is error I'm getting on phone...(LandofOoo = router name) https://t.co/FNQYDNP6lM\n\n(...)",
      "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: No, here is error I'm getting on phone...(LandofOoo = router name) https://t.co/FNQYDNP6lM\n\n- AskSpectrum: Do you have the ability to connect a device via Ethernet, either to the router or directly to the modem?"
    ],
    "corpus": "tweet_summ",
    "ranking": "semantic",
    "chunk": true
  }
}
```

Output:

```json
{
  "query": [
    "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\nCustomer care on Twitter -- Date 2017-10-26 22:00:10+00:00\n\nBrand: Ask Spectrum\n\nAbstract:\nThe customer is complaining that he is facing internet outtage issue.\nThe agent asked the customer tosend an update to the modem to see if it responds and it will reboot if they do that so.",
    "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n- Customer 423147: Experiencing an internet outtage- zipcode 44060\n\n- AskSpectrum: Your modem appears to be online and providing an IP address to an external Linksys router. Has service returned? ^PS\n\n- Customer 423147: No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection\n\n(...)",
    "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection\n\n- AskSpectrum: Would it be OK if I send an update to the modem to see if it responds? It will reboot if I do so. ^PS\n\n- Customer 423147: Yes, that is ok.\n\n(...)",
    "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: Yes, that is ok.\n\n- AskSpectrum: The modem has reset and still shows the connection to the connection to the Linksys router. Are you able to get online now? ^PS\n\n- Customer 423147: No, here is error I'm getting on phone...(LandofOoo = router name) https://t.co/FNQYDNP6lM\n\n(...)",
    "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: No, here is error I'm getting on phone...(LandofOoo = router name) https://t.co/FNQYDNP6lM\n\n- AskSpectrum: Do you have the ability to connect a device via Ethernet, either to the router or directly to the modem?"
  ],
  "docno": [
    "86026c34fe8db0849a9070c0dafa58b1",
    "1eaf47c50e23d44b83d441155046cb55",
    "b32f712ec96129b009d6305e04444263",
    ...
  ],
  "score": [
    0.9950429201126099,
    0.8642614245414734,
    0.851887047290802,
    ...
  ]
}
```

##### Document passage (long query)

To search a document passage in a given collection using a long query, use the `search_doc_long_query` function.
This is useful for searching documents similar to a reference one and using the response suggestions chat function, which usually requires only a portion of a document to see examples of responses.

URL: `http://127.0.0.1:15990/search/doc_chunk_long_query`

Input:

```json
{
  "params": {
    "query": [
      "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\nCustomer care on Twitter -- Date 2017-10-26 22:00:10+00:00\n\nBrand: Ask Spectrum\n\nAbstract:\nThe customer is complaining that he is facing internet outtage issue.\nThe agent asked the customer tosend an update to the modem to see if it responds and it will reboot if they do that so.",
      "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n- Customer 423147: Experiencing an internet outtage- zipcode 44060\n\n- AskSpectrum: Your modem appears to be online and providing an IP address to an external Linksys router. Has service returned? ^PS\n\n- Customer 423147: No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection\n\n(...)",
      "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection\n\n- AskSpectrum: Would it be OK if I send an update to the modem to see if it responds? It will reboot if I do so. ^PS\n\n- Customer 423147: Yes, that is ok.\n\n(...)",
      "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: Yes, that is ok.\n\n- AskSpectrum: The modem has reset and still shows the connection to the connection to the Linksys router. Are you able to get online now? ^PS\n\n- Customer 423147: No, here is error I'm getting on phone...(LandofOoo = router name) https://t.co/FNQYDNP6lM\n\n(...)",
      "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: No, here is error I'm getting on phone...(LandofOoo = router name) https://t.co/FNQYDNP6lM\n\n- AskSpectrum: Do you have the ability to connect a device via Ethernet, either to the router or directly to the modem?"
    ],
    "corpus": "tweet_summ",
    "ranking": "semantic"
  }
}
```

Output:

```json
{
  "query": [
    "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\nCustomer care on Twitter -- Date 2017-10-26 22:00:10+00:00\n\nBrand: Ask Spectrum\n\nAbstract:\nThe customer is complaining that he is facing internet outtage issue.\nThe agent asked the customer tosend an update to the modem to see if it responds and it will reboot if they do that so.",
    "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n- Customer 423147: Experiencing an internet outtage- zipcode 44060\n\n- AskSpectrum: Your modem appears to be online and providing an IP address to an external Linksys router. Has service returned? ^PS\n\n- Customer 423147: No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection\n\n(...)",
    "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection\n\n- AskSpectrum: Would it be OK if I send an update to the modem to see if it responds? It will reboot if I do so. ^PS\n\n- Customer 423147: Yes, that is ok.\n\n(...)",
    "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: Yes, that is ok.\n\n- AskSpectrum: The modem has reset and still shows the connection to the connection to the Linksys router. Are you able to get online now? ^PS\n\n- Customer 423147: No, here is error I'm getting on phone...(LandofOoo = router name) https://t.co/FNQYDNP6lM\n\n(...)",
    "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: No, here is error I'm getting on phone...(LandofOoo = router name) https://t.co/FNQYDNP6lM\n\n- AskSpectrum: Do you have the ability to connect a device via Ethernet, either to the router or directly to the modem?"
  ],
  "docno": [
    "86026c34fe8db0849a9070c0dafa58b1%p3",
    "86026c34fe8db0849a9070c0dafa58b1%p1",
    "86026c34fe8db0849a9070c0dafa58b1%p2",
    ...
  ],
  "score": [
    0.8870396614074707,
    0.8865404248237609,
    0.8849631667137146,
    ...
  ]
}
```

#### Generate snippet

Snippets are query-biased summaries of documents that usually provide a preview of the relevant content in a document for a specific query. 
We provide a generic function encapsulating all snippet generation functionalities and some specific functions to simplify the use of the snippet generation system.

> [!WARNING]
> 
> Do not generate the snippet of too many documents at the same time or the process will take too long.
> For example, in the demo, the search application generates the snippets for 8 documents at a time.

URL: `http://127.0.0.1:15990/snippet`

Input:

```json
{
  "params": {
    "query": "connectivity issue",
    "corpus": "tweet_summ",
    "ranking": "semantic",
    "search_results": {
      "docno": [
        "86026c34fe8db0849a9070c0dafa58b1",
        "8d31903f7989dfb707641538e65fcea3",
        "61c958a7774587c526640bf4ae2642b3",
        ...
      ],
      "score": [
        0.6285992860794067,
        0.6274384260177612,
        0.6134823560714722,
        ...
      ]
    }
  }
}
```

Output:

```json
{
  "docno": [
    "86026c34fe8db0849a9070c0dafa58b1",
    "8d31903f7989dfb707641538e65fcea3",
    "61c958a7774587c526640bf4ae2642b3",
    ...
  ],
  "query": "connectivity issue",
  "score": [
    0.6285992860794067,
    0.6274384260177612,
    0.6134823560714722
  ],
  "summary": [
    "Customer care on Twitter -- Date 2017-10-26 22:00:10+00:00 Brand: Ask Spectrum Abstract: The customer is complaining that he is facing internet outtage issue. The agent asked the customer tosend an update to the modem to see if it responds and it will reboot if they do that so. - Customer 423147: Experiencing an internet outtage- zipcode 44060 - Ask_Spectrum: Your modem appears to be online and providing an IP address to an external Linksys router. Has service returned? ^PS - Customer 423147: No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection - Ask_Spectrum: Would it be OK if I send an update to the modem to see if it responds? It will reboot if I do so. ^PS - Customer 423147: Yes, that is ok. - Ask_Spectrum: The modem has reset and still shows the connection",
    "Customer care on Twitter -- Date 2017-10-27 03:37:02+00:00 Brand: VerizonSupport Abstract: Customer having am issue with internet as his wifi disconnecting issue. Agent receiving a complaint n explaining the issue about wired port connection. - Customer 425638: Raise your hand if you think is the shittiest home Wi-Fi provider to exist!!!! https://t.co/mfLwKevMGC - VerizonSupport: Let's turn your sentiment around! What's going on with your Fios service? ^DDD - Customer 425638: Seems like I can't use my damn Wi-Fi anymore because every other minute my Verizon Wi-Fi is doing this. https://t.co/TdR5eU2Ice - Customer 425638: I can only shut the router off so many times before I give up and get over charged on my 4G for the month. - VerizonSupport: We'll fix it. When you lose connection, can you check if the router's internet light turns red? ^DDD - Customer 425638: This is what I'm seeing. It just shut off.... yet",
    "Customer care on Twitter -- Date 2017-11-22 19:04:54+00:00 Brand: VerizonSupport Abstract: The customer is complaining that even though they have plugged in everything it says they have no coax connection. The agent asked to dm them even the issue persists. - Customer 270787: can you guys help me? my router is plugged in and everything is connected but there's a red globe and https://t.co/w8L8RQFVEC says i have no coax connection. please help - VerizonSupport: We can help! Try a manual reboot. Do so by unplugging the router and unscrewing the coax cable. After 3mins, re-screw the coax cable and plug in the router. This should help! Let us know. ^DDG - Customer 270787: Have done this 30 times but will try again - VerizonSupport: Thank you for trying. Do you have access to your battery backup unit? It is usually a black or white box with blue buttons mounted on",
    ...
  ]
}
```

##### Results

To generate a snippet given the search results obtained from a simple query, use the `generate_snippet` function.
This is useful for providing a preview in a search engine.

URL: `http://127.0.0.1:15990/snippet/generate`

Input:

```json
{
  "params": {
    "query": "connectivity issue",
    "corpus": "tweet_summ",
    "ranking": "semantic",
    "search_results": {
      "docno": [
        "86026c34fe8db0849a9070c0dafa58b1",
        "8d31903f7989dfb707641538e65fcea3",
        "61c958a7774587c526640bf4ae2642b3",
        ...
      ],
      "score": [
        0.6285992860794067,
        0.6274384260177612,
        0.6134823560714722,
        ...
      ]
    }
  }
}
```

Output:

```json
{
  "docno": [
    "86026c34fe8db0849a9070c0dafa58b1",
    "8d31903f7989dfb707641538e65fcea3",
    "61c958a7774587c526640bf4ae2642b3",
    ...
  ],
  "query": "connectivity issue",
  "score": [
    0.6285992860794067,
    0.6274384260177612,
    0.6134823560714722,
    ...
  ],
  "summary": [
    "Customer care on Twitter -- Date 2017-10-26 22:00:10+00:00 Brand: Ask Spectrum Abstract: The customer is complaining that he is facing internet outtage issue. The agent asked the customer tosend an update to the modem to see if it responds and it will reboot if they do that so. - Customer 423147: Experiencing an internet outtage- zipcode 44060 - Ask_Spectrum: Your modem appears to be online and providing an IP address to an external Linksys router. Has service returned? ^PS - Customer 423147: No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection - Ask_Spectrum: Would it be OK if I send an update to the modem to see if it responds? It will reboot if I do so. ^PS - Customer 423147: Yes, that is ok. - Ask_Spectrum: The modem has reset and still shows the connection",
    "Customer care on Twitter -- Date 2017-10-27 03:37:02+00:00 Brand: VerizonSupport Abstract: Customer having am issue with internet as his wifi disconnecting issue. Agent receiving a complaint n explaining the issue about wired port connection. - Customer 425638: Raise your hand if you think is the shittiest home Wi-Fi provider to exist!!!! https://t.co/mfLwKevMGC - VerizonSupport: Let's turn your sentiment around! What's going on with your Fios service? ^DDD - Customer 425638: Seems like I can't use my damn Wi-Fi anymore because every other minute my Verizon Wi-Fi is doing this. https://t.co/TdR5eU2Ice - Customer 425638: I can only shut the router off so many times before I give up and get over charged on my 4G for the month. - VerizonSupport: We'll fix it. When you lose connection, can you check if the router's internet light turns red? ^DDD - Customer 425638: This is what I'm seeing. It just shut off.... yet",
    "Customer care on Twitter -- Date 2017-11-22 19:04:54+00:00 Brand: VerizonSupport Abstract: The customer is complaining that even though they have plugged in everything it says they have no coax connection. The agent asked to dm them even the issue persists. - Customer 270787: can you guys help me? my router is plugged in and everything is connected but there's a red globe and https://t.co/w8L8RQFVEC says i have no coax connection. please help - VerizonSupport: We can help! Try a manual reboot. Do so by unplugging the router and unscrewing the coax cable. After 3mins, re-screw the coax cable and plug in the router. This should help! Let us know. ^DDG - Customer 270787: Have done this 30 times but will try again - VerizonSupport: Thank you for trying. Do you have access to your battery backup unit? It is usually a black or white box with blue buttons mounted on",
    ...
  ]
}
```

##### Results (long query)

To generate a snippet given the search results obtained from a long query, use the `generate_snippet_long_query` function.
This is useful for providing a preview when searching for documents similar to a reference one.

URL: `http://127.0.0.1:15990/snippet/generate_long_query`

Input:

```json
{
  "params": {
    "query": [
      "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\nCustomer care on Twitter -- Date 2017-10-26 22:00:10+00:00\n\nBrand: Ask Spectrum\n\nAbstract:\nThe customer is complaining that he is facing internet outtage issue.\nThe agent asked the customer tosend an update to the modem to see if it responds and it will reboot if they do that so.",
      "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n- Customer 423147: Experiencing an internet outtage- zipcode 44060\n\n- AskSpectrum: Your modem appears to be online and providing an IP address to an external Linksys router. Has service returned? ^PS\n\n- Customer 423147: No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection\n\n(...)",
      "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection\n\n- AskSpectrum: Would it be OK if I send an update to the modem to see if it responds? It will reboot if I do so. ^PS\n\n- Customer 423147: Yes, that is ok.\n\n(...)",
      "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: Yes, that is ok.\n\n- AskSpectrum: The modem has reset and still shows the connection to the connection to the Linksys router. Are you able to get online now? ^PS\n\n- Customer 423147: No, here is error I'm getting on phone...(LandofOoo = router name) https://t.co/FNQYDNP6lM\n\n(...)",
      "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: No, here is error I'm getting on phone...(LandofOoo = router name) https://t.co/FNQYDNP6lM\n\n- AskSpectrum: Do you have the ability to connect a device via Ethernet, either to the router or directly to the modem?"
    ],
    "corpus": "tweet_summ",
    "ranking": "semantic",
    "search_results": {
      "docno": [
        "1eaf47c50e23d44b83d441155046cb55",
        "b32f712ec96129b009d6305e04444263",
        "a0e018a6dbe80300fe73d135471f7a72",
        ...
      ],
      "score": [
        0.8642614245414734,
        0.851887047290802,
        0.8355586171150208,
        ...
      ]
    }
  }
}
```

Output:

```json
{
  "docno": [
    "1eaf47c50e23d44b83d441155046cb55",
    "b32f712ec96129b009d6305e04444263",
    "a0e018a6dbe80300fe73d135471f7a72"
  ],
  "query": [
    "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\nCustomer care on Twitter -- Date 2017-10-26 22:00:10+00:00\n\nBrand: Ask Spectrum\n\nAbstract:\nThe customer is complaining that he is facing internet outtage issue.\nThe agent asked the customer tosend an update to the modem to see if it responds and it will reboot if they do that so.",
    "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n- Customer 423147: Experiencing an internet outtage- zipcode 44060\n\n- AskSpectrum: Your modem appears to be online and providing an IP address to an external Linksys router. Has service returned? ^PS\n\n- Customer 423147: No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection\n\n(...)",
    "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: No- both my modem and router appear to be functioning but none of my internet reliant items (phone, laptop, etc) have internet connection\n\n- AskSpectrum: Would it be OK if I send an update to the modem to see if it responds? It will reboot if I do so. ^PS\n\n- Customer 423147: Yes, that is ok.\n\n(...)",
    "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: Yes, that is ok.\n\n- AskSpectrum: The modem has reset and still shows the connection to the connection to the Linksys router. Are you able to get online now? ^PS\n\n- Customer 423147: No, here is error I'm getting on phone...(LandofOoo = router name) https://t.co/FNQYDNP6lM\n\n(...)",
    "Customer Care Chat 86026c34fe8db0849a9070c0dafa58b1\n\n(...)\n\n- Customer 423147: No, here is error I'm getting on phone...(LandofOoo = router name) https://t.co/FNQYDNP6lM\n\n- AskSpectrum: Do you have the ability to connect a device via Ethernet, either to the router or directly to the modem?"
  ],
  "score": [
    0.8642614245414734,
    0.851887047290802,
    0.8355586171150208,
    ...
  ],
  "summary": [
    "Customer care on Twitter -- Date 2017-11-30 06:01:35+00:00 Brand: Ask Spectrum Abstract: Customer is facing problem with his router connection going offline or down. Agents informs customer about pulling up the schedule. - Customer 120960: Why isn't my router going online????????? Answer that for me. It makes no sense at this point. It already drops connection atleast once a day and now it's been offline for awhile now. Enlighten me. - Ask_Spectrum: Good evening! I would really like to help with your internet services. When you can plz DM your phone number and address to chat? ^DT - Customer 120960: done. - Ask_Spectrum: Thank you for your response. It looks as though there has been some intermittent connectivity with your modem for the last few days. This would require a service appointment to have the matter addressed. We ask that you please send your general availabi... https://t.co/9NuFVZ7J5Z - Customer 120960:",
    "Customer care on Twitter -- Date 2017-10-17 13:03:06+00:00 Brand: Ask Spectrum Abstract: Customer is complaining that they got a message saying that they are valued customer with an offer but the offer was bogus. Agent apologies for the for not providing that offer and also updates that their sales team will be able to give them the most up to date discounts and packages available at the present time. - Customer 502246: Got a message from saying I was a valued customer and with an offer. Offer is bogus! Bait and switch at its worst! - Ask_Spectrum: I am sorry we had not been able to provide that for you. Had you received the offer via USPS or email? ^JR - Customer 502246: USPS - Ask_Spectrum: Hello Ed, I apologize for the unpleasant experience. Can you DM us this offer information so that we can fully investigate it? ^ JMM",
    "Customer care on Twitter -- Date 2017-10-20 21:21:36+00:00 Brand: Ask Spectrum Abstract: Customer is complaning that modem is not connected. Agent is suggests to submit a request to have the power to the lines using URL. - Customer 300144: https://t.co/3kORV8pjnM - Ask_Spectrum: I apologize for the issues. I am showing that we are still working on the construction needed to provide service ... https://t.co/JjtEnCtPbO - Customer 300144: your construction people finished the line work yesterday and in their words \" you are ready for modem installation tomorrow.\". - Customer 300144: words are admitedly harsh. But this is going on 4 months of false promises and pushed back dates. Forgive the rudeness this has been a mess. - Ask_Spectrum: I apologize for the confusion. I can see what I can do to help. Can you please DM us and let me know if the modem... https://t.co/JjtEnCLq3m - Customer 300144: No the",
    ...
  ]
}
```

## Fine-tuning

In the following, we provide instructions on fine-tuning language models on domain-specific data for chatting or searching.

All fine-tuning scripts require configurations provided via YAML files; for further details, refer to the examples in the `./resources/configs/` directory.

### Chat

There are scripts to fine-tune (large) language models on domain-specific data.
The scripts expect the `./src/` directory in the Python path.

In the following, we provide the instructions to fine-tune one of the language models available in the [Transformers library](https://huggingface.co/docs/transformers/index) from [Huggingface]().
Additionally, we provide instructions to monitor the fine-tuning process.

#### Run

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

#### Monitor

It is possible to monitor the fine-tuning process using [Tensorboard](https://www.tensorflow.org/tensorboard).

To connect to a remote server and monitor the fine-tuning process, connect via ssh to your remote machine using a tunnel:

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

### Search

For now, we suggest you train custom search models for ranking using the utilities from the [Sentence-Transformer library](https://www.sbert.net), which is the core of the search services.

## Deployiment

We prepared scripts to deploy Web APIs for chatting and searching using the utilities developed within the scope of this project.
The APIs can be deployed either via [Docker](https://www.docker.com) containers (suggested) or running them manually.
To get insights and examples on how to run these services, please refer to [this project](https://github.com/vincenzo-scotti/me_project/v1.0) using LLM CSTK.

## Acknowledgements

- Vincenzo Scotti: ([vincenzo.scotti@polimi.it](mailto:vincenzo.scotti@polimi.it))
- Mark James Carman: ([mark.carman@polimi.it](mailto:mark.carman@.polimi.it))
