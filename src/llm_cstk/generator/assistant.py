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

    def generate_llm(self, utterances: List[Dict[str, str]]):
        response: str = self._llm.completion(utterances)
        output: Dict[str, str] = {SPEAKER: AI, TEXT: response}

        return output

    def _get_custom_lm_params(self, task: Task, corpus: str) -> Tuple[CustomLM, Optional[str], Dict]:
        custom_lm = self._custom_lm_factory.lm(task, corpus)
        instructions = self._custom_lm_factory.instructions(task, corpus)
        generate_params = self._custom_lm_factory.generate_params(task, corpus)

        return custom_lm, instructions, generate_params

    def generate_custom_lm(self, sample, task: Task, corpus: str, *args, **kwargs) -> str:
        custom_lm, instructions, generate_params = self._get_custom_lm_params(task, corpus)
        sample[INSTRUCTIONS] = instructions
        output = custom_lm.generate(sample, *args, **kwargs, **generate_params)

        return output

    def candidate_responses_llm(
            self,
            utterances: List[Dict[str, str]],
            info: Optional[str] = None,
            candidates: Optional[List[Dict[str, str]]] = None,
            relevant_documents: Optional[List[str]] = None
    ):
        if info is not None:
            utterances.insert(0, {SPEAKER: SYSTEM, TEXT: info})
        instructions = self._llm.instructions.get(CANDIDATE_RESPONSES)
        if instructions is not None:
            utterances.insert(0, {SPEAKER: SYSTEM, TEXT: instructions})
        if candidates is not None and len(candidates) > 0:
            template = self._llm.templates.get(CANDIDATE_RESPONSES)[CANDIDATES]
            candidates = BLOCK_SEP.join(
                template['format'].format(i, example[TEXT]) for i, example in enumerate(candidates, start=1)
            )
            utterances.append({
                SPEAKER: SYSTEM,
                TEXT: f"{template['prefix']}{BLOCK_SEP}{candidates}"
            })
        if relevant_documents is not None and len(relevant_documents) > 0:
            template = self._llm.templates.get(CANDIDATE_RESPONSES)[RELEVANT_DOCS]
            relevant_documents = BLOCK_SEP.join(
                template['format'].format(i, doc) for i, doc in enumerate(relevant_documents, start=1)
            )
            utterances.append({
                SPEAKER: SYSTEM,
                TEXT: f"{template['prefix']}{BLOCK_SEP}{relevant_documents}"
            })

        return self.generate_llm(utterances)

    def candidate_responses_custom_lm(
            self,
            utterances: List[Dict[str, str]],
            speaker: str,
            corpus: str,
            info: Optional[str] = None,
            n_samples: int = 1
    ) -> List[Dict[str, str]]:
        #
        utterances.append({SPEAKER: speaker})
        dialogue: Dict = {INFO: info, UTTERANCES: utterances}
        #
        kwargs = {'output_prefix': True}

        return [
            {SPEAKER: AI, TEXT: self.generate_custom_lm(dialogue, CANDIDATE_RESPONSES, corpus, **kwargs)}
            for _ in range(n_samples)
        ]

    def candidate_responses(
            self,
            model: LM,
            utterances: List[Dict[str, str]],
            speaker: Optional[str] = None,
            corpus: Optional[str] = None,
            info: Optional[str] = None,
            **kwargs
    ) -> List[Dict[str, str]]:
        if model == 'llm':
            return [self.candidate_responses_llm(utterances, info=info, **kwargs)]
        elif model == 'custom_lm':
            return self.candidate_responses_custom_lm(utterances, speaker, corpus, info=info, **kwargs)
        else:
            raise ValueError(
                f"Unknown model type: `{model}`, accepted values are {', '.join(f'{repr(t)}' for t in Scoring)}"
            )

    def info_extraction(
            self,
            utterances: List[Dict[str, str]],
            document: str
    ) -> Dict[str, str]:
        instructions = self._llm.instructions.get(INFO_EXTRACTION)
        if instructions is not None:
            utterances.insert(0, {SPEAKER: SYSTEM, TEXT: instructions})
        template = self._llm.templates.get(INFO_EXTRACTION)[RELEVANT_DOC]
        utterances.append({SPEAKER: SYSTEM, TEXT: template['format'].format(document)})

        return self.generate_llm(utterances)

    def kb_qa(
            self,
            utterances: List[Dict[str, str]],
            relevant_documents: Optional[List[str]] = None
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

        return self.generate_llm(utterances)
