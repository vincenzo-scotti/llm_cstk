import requests
import json

from typing import List, Union

from .language_model import *


class _Chatbot:
    CHATBOT_TYPE: Optional[str] = None
    # Keys
    DOCUMENTS: str = 'docs'
    UTTERANCES: str = 'utterances'
    SPEAKER: str = 'speaker'
    TEXT: str = 'text'
    # Special tokens
    SEP: str = '\n\n'
    BLOCK_SEP: str = '\n\n'
    SPEAKER_SEP: str = ': '
    EOS: str = '\n'
    UTT_SEP: str = '\n'

    def __init__(self, *args, **kwargs):
        # Chat cache
        self._active_chats: Dict[str, Dict[str, Union[List[str], List[Dict[str, str]]]]] = dict()

    @classmethod
    def create_instance(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def add_chat(self, chat_id: str, contextual_data: Optional[List[str]] = None):
        if chat_id not in self._active_chats:
            contextual_data = contextual_data if contextual_data is not None else list()
            self._active_chats[chat_id] = {self.DOCUMENTS: contextual_data, self.UTTERANCES: list()}
        else:
            raise ValueError('Chat already present')

    def remove_chat(self, chat_id: str):
        if chat_id in self._active_chats:
            self._active_chats.pop(chat_id)

    def get_welcome_message(self, chat_id: str) -> Dict[str, str]:
        raise NotImplementedError()

    def generate_response(self, utterance: Dict[str, str], chat_id: str) -> Dict[str, str]:
        raise NotImplementedError()


class Vicuna(_Chatbot):
    # TODO make singleton
    # TODO make start server self managed
    CHATBOT_TYPE: str = 'vicuna'
    # Roles
    SYSTEM: str = 'system'
    ASSISTANT: str = 'assistant'
    USER: str = 'user'
    # Message structure
    CHOICES: str = 'choices'
    MESSAGE: str = 'message'
    MESSAGES: str = 'messages'
    ROLE: str = 'role'
    CONTENT: str = 'content'
    # Requests info
    URL: str = 'http://{}:{}/v1/chat/completions'
    HEADERS: Dict[str, str] = {'accept': 'application/json', 'Content-Type': 'application/json'}

    def __init__(self, *args, address: str = 'localhost', port: int = 8000, **kwargs):
        super().__init__(*args, **kwargs)
        # Llama LLM Server
        self.address: str = address
        self.port: int = port
        # Roles
        self.system = kwargs.get('system', self.SYSTEM)
        self.user = kwargs.get('user', self.SYSTEM)
        self.assistant = kwargs.get('assistant', self.SYSTEM)
        # Prompt
        self.prompt: str = kwargs.get('prompt', '')
        self.welcome_message: Optional[str] = kwargs.get('welcome_message', '')
        self.generate_params: Dict = kwargs.get('generate_params', dict())

    def _prepare_input(self, chat_id: str) -> Dict:
        # Build input
        return {
            self.MESSAGES: [
                {
                    self.ROLE: self.SYSTEM,
                    self.CONTENT: self.prompt.format(self._active_chats[chat_id][self.DOCUMENTS])
                }
            ] + [
                {
                    self.ROLE: self.USER if utterance[self.SPEAKER] == self.user else self.ASSISTANT,
                    self.CONTENT: utterance[self.TEXT]
                }
                for utterance in self._active_chats[chat_id][self.UTTERANCES]
            ],
            **self.generate_params
        }

    def get_welcome_message(self, chat_id: str) -> Dict[str, str]:
        welcome_message = {self.SPEAKER: self.assistant, self.TEXT: self.welcome_message}
        self._active_chats[chat_id][self.UTTERANCES].append(welcome_message)
        request_data = json.dumps(self._prepare_input(chat_id) | {'max_tokens': 1})
        _ = requests.post(self.URL.format(self.address, self.port), headers=self.HEADERS, data=request_data)

        return welcome_message

    def generate_response(self, utterance: Dict[str, str], chat_id: str) -> Dict[str, str]:
        # Add the latest message to chat
        self._active_chats[chat_id][self.UTTERANCES].append({self.SPEAKER: self.user, self.TEXT: utterance[self.TEXT]})
        # Prepare response
        response = {self.SPEAKER: self.assistant, self.TEXT: None}
        # return response
        # Prepare Llama LLM input
        request_data = json.dumps(self._prepare_input(chat_id))
        # Send message to server
        http_response = requests.post(self.URL.format(self.address, self.port), headers=self.HEADERS, data=request_data)
        # Decode output
        if http_response.status_code == 200:
            # Get response text
            response_data: Dict = http_response.json()
            response_str: str = response_data[self.CHOICES][0][self.MESSAGE][self.CONTENT]
            # Update output response
            response[self.TEXT] = response_str
            # Append response to chat history
            self._active_chats[chat_id][self.UTTERANCES].append(response)

        return response
