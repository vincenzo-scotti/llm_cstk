import copy

import pytorch_lightning as pl

import torchmetrics

from transformers import AutoConfig, PretrainedConfig
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, PreTrainedModel
from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding

from submodules.llm_cstk.src.llm_cstk.data.utils import *
from submodules.llm_cstk.src.llm_cstk.data.preparation import sample_to_string

from .metrics import *
from .optim import AdaFactor


class DialogueLM(pl.LightningModule):
    SUBMODULES: List[str] = ['model', 'tokeniser', 'optimiser', 'lr_scheduler', 'generator']

    def __init__(
            self,
            transformer: str,
            device: Optional[torch.device] = None,
            checkpoint_gradient: bool = False,
            ignore_idx: int = IGNORE_IDX,
            compute_lm_metrics: bool = False,
            compute_generator_metrics: bool = False,
            **kwargs
    ):
        super(DialogueLM, self).__init__()
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
        # Configs
        self.model_device: torch.device
        if device is not None:
            self.model_device = device
        else:
            self.model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_gradient: bool = checkpoint_gradient
        # Transformer model
        self.transformer: str = transformer
        self._tokeniser: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.transformer, **self._submodules_params['tokeniser']
        )
        self._language_model: PreTrainedModel
        cfg: PretrainedConfig = AutoConfig.from_pretrained(self.transformer)
        if cfg.is_encoder_decoder:
            self._language_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.transformer, **self._submodules_params['model']
            )
            if self.checkpoint_gradient:
                if hasattr(self._language_model, 'encoder'):
                    self._language_model.encoder.gradient_checkpointing_enable()
                if hasattr(self._language_model, 'decoder'):
                    self._language_model.decoder.gradient_checkpointing_enable()
        else:
            self._language_model = AutoModelForCausalLM.from_pretrained(
                self.transformer, **self._submodules_params['model']
            )
            if self.checkpoint_gradient:
                if hasattr(self._language_model, 'transformer'):
                    self._language_model.transformer.gradient_checkpointing_enable()
                elif hasattr(self._language_model, 'model'):
                    self._language_model.model.gradient_checkpointing_enable()

        # Ignore label ID
        self.ignore_idx: int = ignore_idx
        # Metrics
        self.compute_lm_metrics: bool = compute_lm_metrics
        self._lm_metrics: Optional[Dict[str, torchmetrics.metric.Metric]] = None
        if self.compute_lm_metrics:
            self._lm_metrics = {
                'Perplexity': PPLScore(ignore_idx=self.ignore_idx).to(self.model_device),
            }
        self.compute_generator_metrics: bool = compute_generator_metrics and len(self._submodules_params['generator']) > 0
        self._generator_metrics: Optional[Dict[str, Dict[str, torchmetrics.metric.Metric]]] = None
        if self.compute_generator_metrics:
            self._generator_metrics: Dict[str, Dict[str, torchmetrics.metric.Metric]] = {
                generator_configs_id: {
                    f'BLEU-{n + 1}': BLEUScore(n_gram_size=n+1) for n in range(4)
                } | {
                    'F1': F1Score()
                } | {
                    f'Distinct-{n + 1}': DistinctNScore(normalisation='corpus', n_gram_size=n+1) for n in range(2)
                }
                for generator_configs_id in self._submodules_params['generator']
            }

    def set_n_training_steps(self, n: int):
        if len(self._submodules_params['lr_scheduler']) > 0:
            self._submodules_params['lr_scheduler']['steps_per_epoch'] = int(math.ceil(n))

    def configure_optimizers(self):
        # Build optimiser
        if self._language_model.config.is_encoder_decoder or self.checkpoint_gradient:
            optimiser = AdaFactor(self.parameters(), **self._submodules_params['optimiser'])
        else:
            optimiser = torch.optim.AdamW(self.parameters(), **self._submodules_params['optimiser'])
        # Check whether LR scheduling is required
        if len(self._submodules_params['lr_scheduler']) > 0:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, **self._submodules_params['lr_scheduler'])
            return [optimiser], [{'scheduler': scheduler, 'interval': 'step'}]
        else:
            return optimiser

    def invert_padding_side(self):
        # Used if generating multiple sequences at the same time
        if not self._language_model.config.is_encoder_decoder:
            if self._tokeniser.padding_side == 'right':
                self._submodules_params['tokeniser']['padding_side'] = 'left'
            else:
                self._submodules_params['tokeniser']['padding_side'] = 'right'
            self._tokeniser: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                self.transformer, **self._submodules_params['tokeniser']
            )

    def save(self, path: str):
        # Create tgt path if not exists
        if not os.path.exists(path):
            os.mkdir(path)
        # Make sure model in on CPU
        self.cpu()
        # Save Tokeniser
        self._tokeniser.save_pretrained(path)
        # Save Transformer model
        self._language_model.save_pretrained(path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None):
        # TODO check if loads from different target and source devices
        # Create model instance loading (pre-)trained transformers
        model = DialogueLM(path, device=device)
        # Set in evaluation mode
        model.eval()
        # Move model to device
        model = model.to(model.model_device)

        return model

    def forward(self, src_encoding: BatchEncoding, tgt_encoding: Optional[BatchEncoding] = None) -> torch.tensor:
        # Run transformer model
        logits: torch.tensor
        if self._language_model.config.is_encoder_decoder:
            model_output = self._language_model(
                **src_encoding,
                decoder_input_ids=tgt_encoding.input_ids,
                decoder_attention_mask=tgt_encoding.attention_mask,
                use_cache=not self.checkpoint_gradient
            )
        else:
            model_output = self._language_model(**src_encoding, use_cache=not self.checkpoint_gradient)
        logits = model_output.logits

        return logits

    @staticmethod
    def _get_utterance_str(utterance: Dict) -> str:
        return f'{utterance[SPEAKER]}{SPEAKER_SEP}{utterance[TEXT]}'

    @staticmethod
    def _get_utterance_prefix_str(utterance: Dict) -> str:
        return f'{utterance[SPEAKER]}{SPEAKER_SEP}'.strip()

    def _prepare_generator_input(
            self, sample: Dict, evaluation: bool = False, output_prefix: bool = False, max_new_tokens: int = 0
    ) -> Union[Tuple[str, str], Tuple[str, str, str]]:
        # NOTE: evaluation and output prefix assume at least
        #       one utterance for causal models and
        #       a response for transducer models per sample
        # Make a copy of the sample to avoid corrupting original data
        sample = copy.deepcopy(sample)
        # Init variables
        chunked: bool = False
        output_prefix_str: str = ''
        output_str: Optional[str] = None
        # Prepare target output
        if output_prefix:
            if self._language_model.config.is_encoder_decoder:
                output_prefix_str = self._get_utterance_prefix_str(sample[RESPONSE])
            else:
                output_prefix_str = self._get_utterance_prefix_str(sample[UTTERANCES][-1])
        if not self._language_model.config.is_encoder_decoder:
            if evaluation:
                output_str = self._get_utterance_str(sample[UTTERANCES][-1])
            if evaluation or output_prefix:
                sample[UTTERANCES] = sample[UTTERANCES][:len(sample[UTTERANCES])-1]
        # Prepare input string
        if self._language_model.config.is_encoder_decoder:
            input_str, _ = sample_to_string(sample, False, self._tokeniser, True)
        else:
            input_str = sample_to_string(sample, False, self._tokeniser, False)

        # TODO check for corner cases
        if self._language_model.config.is_encoder_decoder:
            while len(self._tokeniser(input_str)['input_ids']) > self._tokeniser.model_max_length:
                if len(sample[CONTEXT]) > 0:
                    sample[CONTEXT].pop(0)
                    chunked = True
                input_str, _ = sample_to_string(sample, chunked, self._tokeniser, True)
        else:
            while len(self._tokeniser(input_str)['input_ids']) > self._tokeniser.model_max_length - max_new_tokens:
                if len(sample[UTTERANCES]) > 0:
                    sample[UTTERANCES].pop(0)
                    chunked = True
                input_str = sample_to_string(sample, chunked, self._tokeniser, False)
        # Prepare target output for transducer (if required)
        if evaluation and self._language_model.config.is_encoder_decoder:
            output_str = self._get_utterance_str(sample[RESPONSE])
        # Return string(s)
        if evaluation:
            return input_str, output_prefix_str, output_str
        else:
            return input_str, output_prefix_str

    def _prepare_generator_input_batch(
            self, samples: List[Dict], evaluation: bool = False, output_prefix: bool = False, max_new_tokens: int = 0
    ) -> Union[Tuple[List[str], List[str]], Tuple[List[str], List[str], List[str]]]:
        #
        if not evaluation or self._language_model.config.is_encoder_decoder:
            samples_batch = [
                self._prepare_generator_input(
                    sample, evaluation=evaluation, output_prefix=output_prefix, max_new_tokens=max_new_tokens
                )
                for sample in samples
            ]
        else:
            samples_batch = list()
            for sample in samples:
                for idx in range(len(sample[UTTERANCES])):
                    tmp_sample = copy.deepcopy(sample)
                    tmp_sample[UTTERANCES] = tmp_sample[UTTERANCES][:idx + 1]
                    samples_batch.append(
                        self._prepare_generator_input(
                            tmp_sample,
                            evaluation=evaluation,
                            output_prefix=output_prefix,
                            max_new_tokens=max_new_tokens
                        )
                    )
        #
        return tuple(zip(*samples_batch))

    def generate(
            self, sample: Union[List[Dict], Dict], evaluation: bool = False, output_prefix: bool = False, **kwargs
    ) -> Union[List[str], Tuple[List[str], List[str], List[str]], str, Tuple[str, str, str]]:
        # Single sample
        if not isinstance(sample, list):
            return self.generate([sample], evaluation=evaluation, output_prefix=output_prefix, **kwargs)[0]
        # Multiple samples
        if evaluation:
            input_str, output_prefix_str, target_output_str = self._prepare_generator_input_batch(
                sample,
                evaluation=evaluation,
                output_prefix=True,
                max_new_tokens=kwargs.get('max_new_tokens', 0 if self._language_model.config.is_encoder_decoder else 1)
            )
        else:
            input_str, output_prefix_str = self._prepare_generator_input_batch(
                sample,
                evaluation=evaluation,
                output_prefix=output_prefix,
                max_new_tokens=kwargs.get('max_new_tokens', 0 if self._language_model.config.is_encoder_decoder else 1)
            )
            target_output_str = None
        if not self._language_model.config.is_encoder_decoder:
            input_str = [s + self._tokeniser.eos_token for s in input_str]
        generated_output_str = list()
        batch_size = len(sample)
        for idx in range(0, len(input_str), batch_size):
            if self._language_model.config.is_encoder_decoder:
                input_encodings = self._tokeniser(
                    input_str[idx:idx + batch_size],
                    return_tensors='pt',
                    padding=True
                ).to(self.model_device)
                output_encodings = self._tokeniser(
                    [self._tokeniser.pad_token + s for s in output_prefix_str[idx:idx + batch_size]],
                    return_tensors='pt',
                    padding=True,
                    add_special_tokens=False
                ).to(self.model_device)
                generated_output_str += self._tokeniser.batch_decode(
                    self._language_model.generate(
                        **input_encodings,
                        decoder_input_ids=output_encodings.input_ids,
                        decoder_attention_mask=output_encodings.attention_mask,
                        **kwargs
                    ),
                    skip_special_tokens=True
                )
            else:
                input_encodings = self._tokeniser(
                    [
                        s_in + self._tokeniser.eos_token + s_out
                        for s_in, s_out in zip(
                            input_str[idx:idx + batch_size],
                            output_prefix_str[idx:idx + batch_size]
                        )
                    ],
                    return_tensors='pt',
                    padding=True
                ).to(self.model_device)
                input_ids_len = self._tokeniser(
                    input_str[idx:idx + batch_size], return_tensors='pt', padding=True
                ).input_ids.size(1)
                generated_output_str += self._tokeniser.batch_decode(
                    self._language_model.generate(**input_encodings, **kwargs)[:, input_ids_len:],
                    skip_special_tokens=True
                )
        if evaluation:
            return input_str, target_output_str, generated_output_str
        else:
            return generated_output_str

    def training_step(
            self,
            mini_batch: Tuple[
                Union[Tuple[BatchEncoding, torch.tensor], Tuple[BatchEncoding, BatchEncoding, torch.tensor]],
                List[Dict],
                Split
            ],
            mini_batch_idx: int
    ) -> torch.tensor:
        # Unpack mini-batch
        encodings, samples, split = mini_batch
        # Check whether it is an Encoder-Decoder model or a decoder only
        if self._language_model.config.is_encoder_decoder:
            # Unpack the two encodings and the target labels
            src_encodings, tgt_encodings, labels = encodings
            # Compute logits
            logits: torch.tensor = self.forward(src_encodings, tgt_encoding=tgt_encodings)
        else:
            # Unpack the encoding and the target labels
            input_encodings, labels = encodings
            # Compute logits
            logits: torch.tensor = self.forward(input_encodings)
        # Shift logits to exclude the last element
        logits = logits[..., :-1, :].contiguous()
        # shift labels to exclude the first element
        labels = labels[..., 1:].contiguous()
        # Compute LM loss token-wise
        loss: torch.tensor = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=self.ignore_idx
        )
        # Log LM loss
        self.log(f'Loss/{split.capitalize()}', loss)

        return loss

    def _eval_step(
            self,
            mini_batch: Tuple[
                Union[Tuple[BatchEncoding, torch.tensor], Tuple[BatchEncoding, BatchEncoding, torch.tensor]],
                List[Dict],
                Split
            ],
            mini_batch_idx: int
    ) -> torch.tensor:
        # Unpack mini-batch
        encodings, samples, split = mini_batch
        # Check whether it is an Encoder-Decoder model or a decoder only
        if self._language_model.config.is_encoder_decoder:
            # Unpack the two encodings and the target labels
            src_encodings, tgt_encodings, labels = encodings
            # Compute logits
            logits: torch.tensor = self.forward(src_encodings, tgt_encoding=tgt_encodings)
        else:
            # Unpack the encoding and the target labels
            input_encodings, labels = encodings
            # Compute logits
            logits: torch.tensor = self.forward(input_encodings)
        # Shift logits to exclude the last element
        logits = logits[..., :-1, :].contiguous()
        # shift labels to exclude the first element
        labels = labels[..., 1:].contiguous()
        # Compute LM loss token-wise
        loss: torch.tensor = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=self.ignore_idx
        )
        if self.compute_lm_metrics:
            # Iterate over LM metrics
            for metric in self._lm_metrics.values():
                metric.update(logits, labels)
        if self.compute_generator_metrics:
            # TODO add text logging
            # Iterate over generative approaches
            for generator_configs_id in self._generator_metrics:
                input_str, target_outputs, generated_outputs = self.generate(
                    samples, evaluation=True, **self._submodules_params['generator'][generator_configs_id]
                )
                for metric in self._generator_metrics[generator_configs_id].values():
                    metric.update(generated_outputs, target_outputs)
        # Logging to TensorBoard (if installed) by default
        self.log(f'Loss/{split.capitalize()}', loss)

    def validation_step(self, mini_batch, mini_batch_idx):
        return self._eval_step(mini_batch, mini_batch_idx)

    def test_step(self, mini_batch, mini_batch_idx):
        return self._eval_step(mini_batch, mini_batch_idx)

    def _evaluation_epoch_start(self):
        if self.compute_lm_metrics:
            # Iterate over metrics
            for metric in self._lm_metrics.values():
                # Reset metric
                metric.reset()
        if self.compute_generator_metrics:
            # Iterate over metrics
            for generator_configs_id in self._generator_metrics:
                for metric in self._generator_metrics[generator_configs_id].values():
                    # Reset metric
                    metric.reset()

    def on_validation_epoch_start(self):
        return self._evaluation_epoch_start()

    def on_test_epoch_start(self):
        return self._evaluation_epoch_start()

    def _evaluation_epoch_end(self, split: str):
        if self.compute_lm_metrics:
            # Common metrics
            for metric_id, metric in self._lm_metrics.items():
                try:
                    self.log(f'{metric_id}/{split}', metric.compute())
                except ValueError:
                    self.log(f'{metric_id}/{split}', float('nan'))
        if self.compute_generator_metrics:
            # Generator metrics
            for generator_configs_id in self._generator_metrics:
                for metric_id, metric in self._generator_metrics[generator_configs_id].items():
                    try:
                        self.log(f'{metric_id}/{split} ({generator_configs_id})', metric.compute())
                    except ValueError:
                        self.log(f'{metric_id}/{split} ({generator_configs_id})', float('nan'))

    def on_validation_epoch_end(self):
        return self._evaluation_epoch_end('Validation')

    def on_test_epoch_end(self):
        return self._evaluation_epoch_end('Test')

    def enable_metrics(self):
        self.compute_lm_metrics = True
        self.compute_generator_metrics = True and len(self._submodules_params['generator']) > 0
