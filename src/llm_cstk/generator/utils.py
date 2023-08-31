from llm_cstk.model.dialogue_lm import DialogueLM
from llm_cstk.data.utils import *
from llm_cstk.utils.common import *
from llm_cstk.utils.common import _Singleton

# Types

LM: TypeAlias = Literal['llm', 'custom_lm']
CustomLM: TypeAlias = Union[DialogueLM]
Task: TypeAlias = Literal['candidates', 'info_extraction', 'kb_qa']

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
CANDIDATE_RESPONSES: str = 'candidate_responses'
INFO_EXTRACTION: str = 'info_extraction'
KB_QA: str = 'kb_qa'

# Templates
CANDIDATES: str = 'candidates'
RELEVANT_DOCS: str = 'docs'
RELEVANT_DOC: str = 'docs'

MODEL: str = 'model'
GENERATE_PARAMS: str = 'generate_params'
