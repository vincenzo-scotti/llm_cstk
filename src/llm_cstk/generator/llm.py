from urllib.parse import urljoin
import requests
import json

from .utils import *
from .utils import _Singleton


class LLMAPI(_Singleton):
    def __init__(
            self,
            url: str,
            *args,
            generate_params: Optional[Dict[Task, Dict]] = None,
            instructions: Optional[Dict[Task, str]] = None,
            templates: Optional[Dict[Task, Dict]] = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        #
        self.url: str = url
        self.generate_params: Dict[Task, Dict] = generate_params if generate_params is not None else dict()
        self.instructions: Dict[Task, str] = instructions if instructions is not None else dict()
        self.templates: Dict[Task, Dict] = templates if templates is not None else dict()

    def _prepare_completions_input(self, utterances: List[Dict[str, str]], **generate_params):
        return {MESSAGES: [{ROLE: u[SPEAKER], CONTENT: u[TEXT]} for u in utterances], **generate_params}

    @staticmethod
    def _decode_completions_output(http_response: requests.Response) -> Optional[str]:
        if http_response.status_code == 200:
            return http_response.json()[CHOICES][0][MESSAGE][CONTENT]
        else:
            return None

    def completion(self, utterances: List[Dict[str, str]], **generate_params) -> str:
        request_url: str = urljoin(self.url, COMPLETIONS_PATH)
        request_data: str = json.dumps(self._prepare_completions_input(utterances, **generate_params))
        http_response: requests.Response = requests.post(request_url, headers=HEADERS, data=request_data)
        completion: str = self._decode_completions_output(http_response)

        return completion
