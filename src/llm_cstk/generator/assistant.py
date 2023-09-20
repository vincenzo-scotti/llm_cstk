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
            utterances: List[Dict[str, str]],
            task: Optional[Task] = None,
            custom_generate_params: Optional[Dict] = None
    ) -> Dict[str, str]:
        #
        if custom_generate_params is None:
            custom_generate_params = dict()
        #
        response: str = self._llm.completion(utterances, **(self._get_llm_params(task) | custom_generate_params))
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
        generate_params |= custom_generate_params
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

    def _build_user_message_candidate_responses_llm(
            self,
            utterances: List[Dict[str, str]],
            speaker: str,
            info: Optional[str] = None,
            candidates: Optional[List[Dict[str, str]]] = None,
            relevant_documents: Optional[List[str]] = None
    ) -> Dict[str, str]:
        #
        template = self._llm.templates[CANDIDATE_RESPONSES][UTTERANCES]
        dialogue: str = template['sep'].join(
            template['format'].format(utterance[SPEAKER], utterance[TEXT]).strip()
            for utterance in utterances[:template.get('n_ctx', len(utterances))]
        )
        if len(utterances) > template.get('n_ctx', len(utterances)):
            dialogue = f"{ELLIPS}{template['sep']}{dialogue}"
        if info is not None:
            dialogue = f"{info}{template['sep']}{dialogue}"
        #
        template = self._llm.templates[CANDIDATE_RESPONSES][DIALOGUE]
        user_message: str = template['format'].format(dialogue, speaker)
        #
        if candidates is not None and len(candidates) > 0:
            template = self._llm.templates[CANDIDATE_RESPONSES][CANDIDATES]
            candidates = BLOCK_SEP.join(
                template['format'].format(i, example[TEXT]) for i, example in enumerate(candidates, start=1)
            )
            user_message = f"{user_message}{BLOCK_SEP}{template['prefix']}{BLOCK_SEP}{candidates}"
        #
        if relevant_documents is not None and len(relevant_documents) > 0:
            template = self._llm.templates[CANDIDATE_RESPONSES][RELEVANT_DOCS]
            relevant_documents = BLOCK_SEP.join(
                template['format'].format(i, doc) for i, doc in enumerate(relevant_documents, start=1)
            )
            user_message = f"{user_message}{BLOCK_SEP}{template['prefix']}{BLOCK_SEP}{relevant_documents}"

        return {SPEAKER: USER, TEXT: user_message}

    def _build_message_pair_candidate_responses_llm(
            self, sample: Dict[str, Union[str, List[Dict[str, str]]]]
    ) -> List[Dict[str, str]]:
        sample = copy.deepcopy(sample)
        if RESPONSE in sample:
            response = sample.pop(RESPONSE)
        else:
            response = sample[UTTERANCES].pop(len(sample[UTTERANCES]) - 1)
        user_message = self._build_user_message_candidate_responses_llm(
            sample[UTTERANCES], response[SPEAKER], info=sample.get(INFO)
        )
        assistant_response = {SPEAKER: AI, TEXT: response[TEXT]}

        return [user_message, assistant_response]

    def candidate_responses_llm(
            self,
            utterances: List[Dict[str, str]],
            speaker: str,
            corpus: str,
            info: Optional[str] = None,
            examples: Optional[List[Dict[str, Union[str, List[str]]]]] = None,
            candidates: Optional[List[Dict[str, str]]] = None,
            relevant_documents: Optional[List[str]] = None,
            custom_generate_params: Optional[Dict] = None,
            n_samples: int = 1
    ) -> List[Dict[str, str]]:
        # See https://openai.com/blog/gpt-4-api-general-availability in the "few-shot learning" example
        dialogue = list()
        instructions = self._llm.instructions.get(CANDIDATE_RESPONSES)
        if instructions is not None:
            utterances.insert(0, {SPEAKER: SYSTEM, TEXT: instructions})
        if examples is not None:
            for example in examples:
                dialogue += self._build_message_pair_candidate_responses_llm(example)
        dialogue.append(self._build_user_message_candidate_responses_llm(
            utterances, speaker, info=info, candidates=candidates, relevant_documents=relevant_documents
        ))

        return [
            self.generate_llm(dialogue, task=CANDIDATE_RESPONSES, custom_generate_params=custom_generate_params)
            for _ in range(n_samples)
        ]

    def candidate_responses_custom_lm(
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
                dialogue, CANDIDATE_RESPONSES, corpus, custom_generate_params=custom_generate_params
            )
            for _ in range(n_samples)
        ]

    def candidate_responses(
            self,
            model: LM, *args, **kwargs
    ) -> List[Dict[str, str]]:
        if model == 'llm':
            return self.candidate_responses_llm(*args, **kwargs)
        elif model == 'custom_lm':
            return self.candidate_responses_custom_lm(*args, **kwargs)
        else:
            raise ValueError(
                f"Unknown model type: `{model}`, accepted values are {', '.join(f'{repr(t)}' for t in LM)}"
            )

    def info_extraction(
            self,
            utterances: List[Dict[str, str]],
            document: str,
            custom_generate_params: Optional[Dict] = None
    ) -> Dict[str, str]:
        instructions = self._llm.instructions.get(INFO_EXTRACTION)
        if instructions is not None:
            utterances.insert(0, {SPEAKER: SYSTEM, TEXT: instructions})
        template = self._llm.templates.get(INFO_EXTRACTION)[RELEVANT_DOC]
        utterances.append({SPEAKER: SYSTEM, TEXT: template['format'].format(document)})

        return self.generate_llm(utterances, custom_generate_params=custom_generate_params)

    def kb_qa(
            self,
            utterances: List[Dict[str, str]],
            relevant_documents: Optional[List[str]] = None,
            custom_generate_params: Optional[Dict] = None
    ) -> Dict[str, str]:
        instructions = self._llm.instructions.get(KB_QA)
        if instructions is not None:
            utterances.insert(0, {SPEAKER: SYSTEM, TEXT: instructions})
        if relevant_documents is not None and len(relevant_documents) > 0:
            template = self._llm.templates.get(KB_QA)[RELEVANT_DOCS]
            utterances.append({
                SPEAKER: SYSTEM,
                TEXT: f"{template['prefix']}{BLOCK_SEP}"
                      f"{BLOCK_SEP.join(template['format'].format(i, doc) for i, doc in enumerate(relevant_documents, start=1))}"
            })

        return self.generate_llm(utterances, custom_generate_params=custom_generate_params)
