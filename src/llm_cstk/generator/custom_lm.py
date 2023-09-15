from .utils import *
from .utils import _Singleton


class CustomLMFactory(_Singleton):
    def __init__(self, configs: Dict[Task, Dict[str, Dict]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        #
        self.configs: Dict[Task, Dict[str, Dict]] = configs
        #
        self._transformer_lm_cache: Dict[str, CustomLM] = dict()

    def lm(self, task: Task, corpus: str) -> CustomLM:
        if self.configs[task][corpus][MODEL] not in self._transformer_lm_cache:
            if task == CANDIDATE_RESPONSES:
                transformer_lm = DialogueLM.load(
                    self.configs[task][corpus][MODEL], **self.configs[task][corpus][CONFIGS]
                )
            else:
                raise ValueError(
                    f"Unknown task type: `{task}`, accepted values are {', '.join(f'{repr(t)}' for t in Task)}"
                )

            self._transformer_lm_cache[self.configs[task][corpus][MODEL]] = transformer_lm

        return self._transformer_lm_cache[self.configs[task][corpus][MODEL]]

    def instructions(self, task: Task, corpus: str) -> Optional[str]:
        return self.configs[task][corpus].get(INSTRUCTIONS)

    def generate_params(self, task: Task, corpus: str) -> Dict:
        return self.configs[task][corpus].get(GENERATE_PARAMS, dict())
