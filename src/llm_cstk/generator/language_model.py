import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel

from typing import Dict, Optional


class _LMInterface:
    def __init__(self, lm_model: str, device: Optional[torch.device] = None, **kwargs):
        # Prepare kwargs
        self._submodules_params: Dict[str, Dict] = dict()
        for param_id, param_val in kwargs.items():
            if "__" not in param_id:
                raise ValueError(
                    f"Parameters for the submodules must be passed in the form 'submodule__parameter', "
                    f"received parameter with name '{param_id}'."
                )
            module, param = param_id.split("__", 1)
            if module not in self._submodules_params:
                self._submodules_params[module] = dict()
            self._submodules_params[module][param] = param_val
        #
        self.lm_model: str = lm_model
        self.device: torch.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        # Tokenizer and LM model
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.lm_model, **self._submodules_params.get('model', dict())
        )
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.lm_model, **self._submodules_params.get('model', dict())
        ).eval().to(self.device)
        # Caches
        self._lm_cache: Dict = dict()

    def _load_model(self):
        raise NotImplementedError()

    def process(self, text: str, context: str):
        # Retrieve cache
        cache = self._lm_cache.pop(hash(context), None)
        # Encode input
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids.to(self.device)
        # Get updated cache
        self._lm_cache[hash(f'{context}{text}')] = self.lm_model.transformer(
            input_ids=input_ids, past_key_values=cache
        )  # TODO rework to take into account attention mask
