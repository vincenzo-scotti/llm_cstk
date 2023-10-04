from llm_cstk.model.dialogue_lm import DialogueLM
from llm_cstk.data.utils import *
from llm_cstk.utils.common import *
from llm_cstk.utils.common import _Singleton

# Types

LM: TypeAlias = Literal['llm', 'custom_lm']
CustomLM: TypeAlias = Union[DialogueLM]
Task: TypeAlias = Literal['response_suggestion', 'info_extraction', 'kb_qa']

# Constants

# Roles
SYSTEM: str = 'system'
AI: str = 'assistant'
USER: str = 'user'

# Message structure
CHOICES: str = 'choices'
MESSAGE: str = 'message'
MESSAGES: str = 'messages'
ROLE: str = 'role'
CONTENT: str = 'content'

# Requests info
HEADERS: Dict[str, str] = {'accept': 'application/json', 'Content-Type': 'application/json'}
COMPLETIONS_PATH: str = '/v1/chat/completions'

# Tasks
RESPONSE_SUGGESTION: str = 'response_suggestion'
INFO_EXTRACTION: str = 'info_extraction'
KB_QA: str = 'kb_qa'

# Templates
DIALOGUE: str = 'dialogue'
CANDIDATES: str = 'candidates'
RELEVANT_DOCS: str = 'docs'
RELEVANT_DOC: str = 'doc'

MODEL: str = 'model'
CONFIGS: str = 'configs'
GENERATE_PARAMS: str = 'generate_params'
