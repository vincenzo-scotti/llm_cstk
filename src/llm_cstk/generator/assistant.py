import copy

from .llm import LLMAPI
from .custom_lm import CustomLMFactory
from .utils import *
from .utils import _Singleton


class AIAssistant(_Singleton):
    SUBMODULES: List[str] = ['llm', 'custom_lm_factory']

    def __init__(self, **kwargs):
        super().__init__()
        # Prepare kwargs
        self._submodules_params: Dict[str, Dict] = {key: dict() for key in self.SUBMODULES}
        for param_id, param_val in kwargs.items():
            if "__" not in param_id:
                raise ValueError(
                    f"Parameters for the submodules must be passed in the form 'submodule__parameter', "
                    f"received parameter with name '{param_id}'."
                )
            module, param = param_id.split("__", 1)
            self._submodules_params[module][param] = param_val
        # LLM Chatbot
        self._llm: LLMAPI = LLMAPI.load(**self._submodules_params['llm'])
        self._custom_lm_factory = CustomLMFactory.load(**self._submodules_params['custom_lm_factory'])

    @classmethod
    def load(cls, *args, **kwargs):
        for submodule in cls.SUBMODULES:
            if submodule in kwargs:
                configs = kwargs.pop(submodule)
                for k, v in configs.items():
                    kwargs[f'{submodule}__{k}'] = v
        return super().load(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def _get_llm_params(self, task: Optional[Task]) -> Dict:
        if task is None:
            return dict()
        else:
            return self._llm.generate_params.get(task, dict())

    def generate_llm(
            self,
            sample: Union[List[Dict[str, str]], str],
            task: Optional[Task] = None,
            custom_generate_params: Optional[Dict] = None,
            approach: LLMCompletionApproach = CHAT_COMPLETION
    ) -> Dict[str, str]:
        #
        if custom_generate_params is None:
            custom_generate_params = dict()
        #
        if approach == PLAIN_COMPLETION:
            response = self._llm.completion(sample, **(self._get_llm_params(task) | custom_generate_params))
        elif approach == CHAT_COMPLETION:
            response = self._llm.chat_completion(sample, **(self._get_llm_params(task) | custom_generate_params))
        else:
            raise ValueError(
                f"Unknown LLM completion approach: `{approach}`, "
                f"accepted values are {', '.join(f'{repr(t)}' for t in LLMCompletionApproach)}"
            )
        output: Dict[str, str] = {SPEAKER: AI, TEXT: response}

        return output

    def _get_custom_lm_params(self, task: Task, corpus: str) -> Tuple[CustomLM, Optional[str], Dict]:
        custom_lm = self._custom_lm_factory.lm(task, corpus)
        instructions = self._custom_lm_factory.instructions(task, corpus)
        generate_params = self._custom_lm_factory.generate_params(task, corpus)

        return custom_lm, instructions, generate_params

    def generate_custom_lm(
            self, sample, task: Task, corpus: str, *args, custom_generate_params: Optional[Dict] = None, **kwargs
    ) -> Dict[str, str]:
        custom_lm, instructions, generate_params = self._get_custom_lm_params(task, corpus)
        sample[INSTRUCTIONS] = instructions
        response: str = custom_lm.generate(sample, *args, **kwargs, **(generate_params | custom_generate_params))
        output: Dict[str, str] = {SPEAKER: AI, TEXT: response}

        return output

    def generate(self, model: LM, *args, **kwargs) -> Dict[str, str]:
        if model == 'llm':
            return self.generate_llm(*args, **kwargs)
        elif model == 'custom_lm':
            return self.generate_custom_lm(*args, **kwargs)
        else:
            raise ValueError(
                f"Unknown model type: `{model}`, accepted values are {', '.join(f'{repr(t)}' for t in LM)}"
            )

    def _prepare_dialogue_response_suggestion_llm(
            self,
            utterances: List[Dict[str, str]],
            speaker: Optional[str] = None,
            info: Optional[str] = None,
            candidates: Optional[List[Dict[str, str]]] = None,
            relevant_documents: Optional[List[str]] = None
    ) -> str:
        #
        template = self._llm.templates[RESPONSE_SUGGESTION][UTTERANCES]
        dialogue: List[str] = list()
        dialogue.extend(
            template['format'].format(utterance[SPEAKER], utterance[TEXT]).strip()
            for utterance in utterances[-template.get('n_ctx', len(utterances)):]
        )
        #
        if len(utterances) > template.get('n_ctx', len(utterances)):
            dialogue.insert(0, ELLIPS)
        if info is not None:
            dialogue.insert(0, info)
        #
        if candidates is not None and len(candidates) > 0:
            template_ = self._llm.templates[RESPONSE_SUGGESTION][CANDIDATES]
            candidates = BLOCK_SEP.join(
                template_['format'].format(i, example[TEXT]) for i, example in enumerate(candidates, start=1)
            )
            dialogue.append(template['format'].format(SYSTEM, candidates))
        #
        if relevant_documents is not None and len(relevant_documents) > 0:
            template_ = self._llm.templates[RESPONSE_SUGGESTION][RELEVANT_DOCS]
            relevant_documents = BLOCK_SEP.join(
                template_['format'].format(i, doc) for i, doc in enumerate(relevant_documents, start=1)
            )
            dialogue.append(template['format'].format(SYSTEM, relevant_documents))
        #
        if speaker is not None:
            template_ = self._llm.templates[RESPONSE_SUGGESTION][PROMPT]
            dialogue.append(template_['format'].format(speaker))

        return template['sep'].join(dialogue)

    def response_suggestion_llm(
            self,
            utterances: List[Dict[str, str]],
            speaker: str,
            corpus: str,
            info: Optional[str] = None,
            examples: Optional[List[Dict[str, Union[str, List[Dict[str, str]]]]]] = None,
            candidates: Optional[List[Dict[str, str]]] = None,
            relevant_documents: Optional[List[str]] = None,
            ask_query: bool = False,
            custom_generate_params: Optional[Dict] = None,
            n_samples: int = 1
    ) -> List[Dict[str, str]]:
        # See https://openai.com/blog/gpt-4-api-general-availability in the "few-shot learning" example
        dialogue = list()
        instructions = self._llm.instructions.get(RESPONSE_SUGGESTION)
        if instructions is not None:
            template = self._llm.templates.get(RESPONSE_SUGGESTION)[UTTERANCES]
            dialogue.append(template['format'].format(SYSTEM, instructions))  # [{SPEAKER: SYSTEM, TEXT: instructions}]
        if examples is not None:
            for example in examples:
                dialogue.append(
                    self._prepare_dialogue_response_suggestion_llm(example[UTTERANCES], info=example.get(INFO))
                )
        if ask_query:
            template = self._llm.templates.get(RESPONSE_SUGGESTION)[QUERY]
            utterances.append({SPEAKER: SYSTEM, TEXT: template['message']})
        dialogue.append(self._prepare_dialogue_response_suggestion_llm(
            utterances,
            speaker=speaker,
            info=info,
            candidates=candidates if not ask_query else None,
            relevant_documents=relevant_documents if not ask_query else None
        ))
        template = self._llm.templates.get(RESPONSE_SUGGESTION)[DIALOGUE]
        dialogue = template['sep'].join(dialogue)

        return [
            self.generate_llm(
                dialogue,
                task=RESPONSE_SUGGESTION,
                custom_generate_params=custom_generate_params,
                approach=PLAIN_COMPLETION
            )
            for _ in range(n_samples)
        ]

    def response_suggestion_custom_lm(
            self,
            utterances: List[Dict[str, str]],
            speaker: str,
            corpus: str,
            info: Optional[str] = None,
            custom_generate_params: Optional[Dict] = None,
            n_samples: int = 1
    ) -> List[Dict[str, str]]:
        if custom_generate_params is None:
            custom_generate_params = {'output_prefix': True}
        else:
            custom_generate_params['output_prefix'] = True
        #
        utterances.append({SPEAKER: speaker})
        dialogue: Dict = {INFO: info, UTTERANCES: utterances}

        return [
            self.generate_custom_lm(
                dialogue, RESPONSE_SUGGESTION, corpus, custom_generate_params=custom_generate_params
            )
            for _ in range(n_samples)
        ]

    def response_suggestion(
            self,
            model: LM, *args, **kwargs
    ) -> List[Dict[str, str]]:
        if model == 'llm':
            return self.response_suggestion_llm(*args, **kwargs)
        elif model == 'custom_lm':
            return self.response_suggestion_custom_lm(*args, **kwargs)
        else:
            raise ValueError(
                f"Unknown model type: `{model}`, accepted values are {', '.join(f'{repr(t)}' for t in LM)}"
            )

    def _prepare_dialogue_assistant_chat_llm(self, utterances: List[Dict[str, str]], task: Task) -> str:
        #
        template = self._llm.templates[task][UTTERANCES]
        dialogue: List[str] = list()
        #
        if INFO_EXTRACTION in self._llm.instructions:
            template_ = self._llm.templates[task][INSTRUCTIONS]
            dialogue.append(template_['format'].format(utterances.pop(0)[TEXT]))
        #
        for utterance in utterances:
            dialogue.append(template['formats'][utterance[SPEAKER]].format(utterance[TEXT]))
        #
        dialogue.append(template['suffix'])

        return template['sep'].join(dialogue)

    def info_extraction(
            self,
            utterances: List[Dict[str, str]],
            document: str,
            custom_generate_params: Optional[Dict] = None
    ) -> Dict[str, str]:
        template = self._llm.templates.get(INFO_EXTRACTION)[RELEVANT_DOC]
        utterances.insert(0, {SPEAKER: SYSTEM, TEXT: template['format'].format(document)})
        instructions = self._llm.instructions.get(INFO_EXTRACTION)
        if instructions is not None:
            utterances.insert(0, {SPEAKER: SYSTEM, TEXT: instructions})
        #
        dialogue = self._prepare_dialogue_assistant_chat_llm(utterances, INFO_EXTRACTION)

        return self.generate_llm(
            dialogue,
            task=INFO_EXTRACTION,
            custom_generate_params=custom_generate_params,
            approach=PLAIN_COMPLETION
        )

    def kb_qa(
            self,
            utterances: List[Dict[str, str]],
            relevant_documents: Optional[List[str]] = None,
            ask_query: bool = False,
            custom_generate_params: Optional[Dict] = None
    ) -> Dict[str, str]:
        instructions = self._llm.instructions.get(KB_QA)
        if instructions is not None:
            utterances.insert(0, {SPEAKER: SYSTEM, TEXT: instructions})
        if ask_query:
            template = self._llm.templates.get(KB_QA)[QUERY]
            utterances.append({
                SPEAKER: SYSTEM,
                TEXT: template['message']
            })
        elif relevant_documents is not None and len(relevant_documents) > 0:
            template = self._llm.templates.get(KB_QA)[RELEVANT_DOCS]
            utterances.append({
                SPEAKER: SYSTEM,
                TEXT: f"{template['prefix']}{BLOCK_SEP}"
                      f"{BLOCK_SEP.join(template['format'].format(i, doc) for i, doc in enumerate(relevant_documents, start=1))}"
            })
        #
        dialogue = self._prepare_dialogue_assistant_chat_llm(utterances, KB_QA)

        return self.generate_llm(
                dialogue,
                task=KB_QA,
                custom_generate_params=custom_generate_params,
                approach=PLAIN_COMPLETION
            )
