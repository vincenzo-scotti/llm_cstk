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
            generate_params: Optional[Dict] = None,
            instructions: Optional[Dict[str, str]] = None,
            templates: Optional[Dict[str, str]] = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        #
        self.url: str = url
        self.generate_params: Dict = generate_params if generate_params is not None else dict()
        self.instructions: Dict = instructions if instructions is not None else dict()
        self.templates: Dict = templates if templates is not None else dict()

    def _prepare_completions_input(self, utterances: List[Dict[str, str]]):
        return {
            MESSAGES: [{ROLE: u[SPEAKER], CONTENT: u[TEXT]} for u in utterances], **self.generate_params
        }

    @staticmethod
    def _decode_completions_output(http_response: requests.Response) -> Optional[str]:
        if http_response.status_code == 200:
            return http_response.json()[CHOICES][0][MESSAGE][CONTENT]
        else:
            return None

    def completion(self, utterances: List[Dict[str, str]]) -> str:
        request_url: str = urljoin(self.url, COMPLETIONS_PATH)
        request_data: str = json.dumps(self._prepare_completions_input(utterances))
        http_response: requests.Response = requests.post(request_url, headers=HEADERS, data=request_data)
        completion: str = self._decode_completions_output(http_response)

        return completion
