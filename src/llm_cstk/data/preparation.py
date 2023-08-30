from transformers import PreTrainedTokenizer
from .augmentation import data_augmentation
import copy
from .utils import *


def sample_to_string(
        sample: Dict, chunk: bool, tokeniser: PreTrainedTokenizer, encoder_decoder: bool
) -> Union[str, Tuple[str, str]]:
    #
    instructions: Optional[str] = None
    if INSTRUCTIONS in sample:
        instructions = sample[INSTRUCTIONS]
    #
    info: Optional[str] = None
    if INFO in sample and sample[INFO] is not None and sample[INFO] != UNK:
        info = sample[INFO]
    #
    if encoder_decoder:
        context: str = UTTERANCES_SEP.join(
            f'{utterance[SPEAKER]}{SPEAKER_SEP}{utterance[TEXT]}' for utterance in sample[CONTEXT]
        ) if len(sample[CONTEXT]) > 0 else str(None)
        response: str = f'{sample[RESPONSE][SPEAKER]}{SPEAKER_SEP}{sample[RESPONSE][TEXT]}'
        #
        if chunk:
            src_str = f"{CONTEXT_ID}{SEP}{ELLIPS}{UTTERANCES_SEP}{context}{BLOCK_SEP}{RESPONSE_ID}"
        else:
            src_str = f"{CONTEXT_ID}{SEP}{context}{BLOCK_SEP}{RESPONSE_ID}"
        #
        if info is not None:
            src_str = f"{INFO_ID}{SEP}{info}{BLOCK_SEP}{src_str}"
        if instructions is not None:
            src_str = f'{instructions}{BLOCK_SEP}{src_str}'
        #
        tgt_str = response
        #
        sample_str = (src_str, tgt_str)
    else:
        dialogue: str = tokeniser.eos_token.join(
            f'{utterance[SPEAKER]}{SPEAKER_SEP}{utterance[TEXT]}' for utterance in sample[UTTERANCES]
        )
        sample_str = f'{ELLIPS}{tokeniser.eos_token}{dialogue}' if chunk else dialogue
        #
        if info is not None:
            sample_str = f'{SYSTEM_ID}{SPEAKER_SEP}{info}{tokeniser.eos_token}{sample_str}'
        if instructions is not None:
            sample_str = f'{SYSTEM_ID}{SPEAKER_SEP}{instructions}{tokeniser.eos_token}{sample_str}'

    return sample_str


def _check_fit_in_token_window(
        sample_str:  Union[str, Tuple[str, str]], tokeniser: PreTrainedTokenizer, encoder_decoder: bool
) -> Union[bool, Tuple[bool, bool]]:
    #
    if encoder_decoder:
        src_str, tgt_str = sample_str
        fitting = (
            len(tokeniser(src_str)['input_ids']) < tokeniser.model_max_length,
            len(tokeniser(tgt_str)['input_ids']) < tokeniser.model_max_length
        )
    else:
        fitting = len(tokeniser(sample_str)['input_ids']) < tokeniser.model_max_length

    return fitting


def _fit_in_token_window(
        sample: Dict, tokeniser: PreTrainedTokenizer, encoder_decoder: bool
) -> Tuple[Dict, bool]:
    # TODO manage better long contexts
    # Iteratively pop from context
    chunked: bool = False
    sample_str = sample_to_string(sample, chunked, tokeniser, encoder_decoder)
    if encoder_decoder:
        while not _check_fit_in_token_window(sample_str, tokeniser, encoder_decoder)[0]:
            if len(sample[CONTEXT]) > 0:
                sample[CONTEXT].pop(0)
                chunked = True
            sample_str = sample_to_string(sample, chunked, tokeniser, encoder_decoder)
    else:
        while not _check_fit_in_token_window(sample_str, tokeniser, encoder_decoder):
            if len(sample[UTTERANCES]) > 0:
                sample[UTTERANCES].pop(0)
                chunked = True
            sample_str = sample_to_string(sample, chunked, tokeniser, encoder_decoder)

    return sample, chunked


def prepare_sample(
        sample: Dict,
        tokeniser: PreTrainedTokenizer,
        augmentation: bool = False,
        encoder_decoder: bool = True
) -> Union[str, Tuple[str, str]]:  # TODO automate encoder decoder control
    # Make a copy of the sample to avoid corrupting original data
    sample = copy.deepcopy(sample)
    # Possibly apply data augmentation
    if augmentation:
        sample = data_augmentation(sample, encoder_decoder)
    # Make sure that sample fits in context
    sample, chunked = _fit_in_token_window(sample, tokeniser, encoder_decoder)
    # Convert to string
    sample_str: Union[str, Tuple[str, str]] = sample_to_string(sample, chunked, tokeniser, encoder_decoder)

    return sample_str
