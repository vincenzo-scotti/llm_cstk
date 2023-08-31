import string
from .utils import *


ALTERNATIVE_USR_IDS: Set[str] = {'Speaker', 'Person', 'User'}
ALTERNATIVE_SYS_IDS: Set[str] = {'System'}

ALTERNATIVE_ID_KEYS: Set[Tuple[str]] = {tuple(string.ascii_uppercase), tuple(range(32))}


# TODO add other augmentations
#  Ideas:
#  - Data cleaning (fix grammar)
#  - Translation (to make it robust for other languages)


def _change_ids(sample: Dict, encoder_decoder: bool = True) -> Dict:
    # Get current sample ids
    usr_ids: List[str] = list(set(
        utterance[SPEAKER]
        for utterance in (sample[CONTEXT] + [sample[RESPONSE]] if encoder_decoder else sample[UTTERANCES])
        if utterance[SPEAKER] is not None and not utterance[SYSTEM_FLAG]
    ))
    random.shuffle(usr_ids)
    sys_ids: List[str] = list(set(
        utterance[SPEAKER]
        for utterance in (sample[CONTEXT] + [sample[RESPONSE]] if encoder_decoder else sample[UTTERANCES])
        if utterance[SPEAKER] is not None and utterance[SYSTEM_FLAG]
    ))
    random.shuffle(sys_ids)
    # Get alternative IDS
    ids_alt_key = random.choice(list(ALTERNATIVE_ID_KEYS))
    usr_ids_alt_basename = random.choice(list(ALTERNATIVE_USR_IDS))
    usr_ids_alt: List[str] = [usr_ids_alt_basename] 
    if len(usr_ids) > 1:
        usr_ids_alt = [f'{usr_ids_alt_basename} {key}' for key in ids_alt_key[:len(usr_ids)]]
    sys_ids_alt_basename = random.choice(list(ALTERNATIVE_SYS_IDS))
    sys_ids_alt: List[str] = [sys_ids_alt_basename]
    if len(sys_ids) > 1:
        sys_ids_alt = [f'{sys_ids_alt_basename} {key}' for key in ids_alt_key[:len(sys_ids)]]
    # Generate ids mappings
    ids_mapping: Dict[str, str] = {
        usr_id: usr_id_alt for usr_id, usr_id_alt in zip(usr_ids, usr_ids_alt)
    } | {
        sys_id: sys_id_alt for sys_id, sys_id_alt in zip(sys_ids, sys_ids_alt)
    }
    # Apply mapping
    if encoder_decoder:
        for utterance in sample[CONTEXT]:
            utterance[SPEAKER] = ids_mapping.get(utterance[SPEAKER], utterance[SPEAKER])
        sample[RESPONSE][SPEAKER] = ids_mapping.get(sample[RESPONSE][SPEAKER], sample[RESPONSE][SPEAKER])
    else:
        for utterance in sample[UTTERANCES]:
            utterance[SPEAKER] = ids_mapping.get(utterance[SPEAKER], utterance[SPEAKER])

    return sample


def _change_casing(sample: Dict, encoder_decoder: bool = True) -> Dict:
    # Decide whether to apply upper-casing or lower-casing
    upper: bool = random.choice((True, False))
    # Apply transformation
    if encoder_decoder:
        for utterance in sample[CONTEXT]:
            utterance[TEXT] = utterance[TEXT].upper() if upper else utterance[TEXT].lower()
        sample[RESPONSE][TEXT] = sample[RESPONSE][TEXT].upper() if upper else sample[RESPONSE][TEXT].lower()
    else:
        for utterance in sample[UTTERANCES]:
            utterance[TEXT] = utterance[TEXT].upper() if upper else utterance[TEXT].lower()

    return sample


def _drop_preamble(sample: Dict, encoder_decoder: bool = True) -> Dict:
    if INFO in sample:
        sample.pop(INFO)

    return sample


def _change_instructions(sample, encoder_decoder: bool = True):
    if INSTRUCTIONS_ALTERNATIVES in sample and all(isinstance(alt, list) for alt in sample[INSTRUCTIONS_ALTERNATIVES]):
        instructions = ''
        for alt in sample[INSTRUCTIONS_ALTERNATIVES]:
            instructions = (instructions + random.choice(alt)).strip()
        sample[INSTRUCTIONS] = instructions if len(instructions) > 0 else None

    return sample


METADATA_TRANSFORMATIONS: List[Callable] = [_change_instructions]
TRANSFORMATIONS: List[Callable] = [_change_ids, _change_casing, _drop_preamble]
TRANSFORMATION_FLAGS: List[bool] = [True] * len(TRANSFORMATIONS) + [False] * (len(TRANSFORMATIONS) - 1)


def data_augmentation(sample: Dict, encoder_decoder: bool = True) -> Dict:
    # Appy forced transformations:
    for transform_fn in METADATA_TRANSFORMATIONS:
        sample = transform_fn(sample, encoder_decoder)
    # Decide whether to apply augmentation or not
    augment: bool = random.choice((True, False))
    # Do augmentation (if required)
    if augment:
        # Randomly select which transformations to apply
        transformation_flags = random.choices(TRANSFORMATION_FLAGS, k=len(TRANSFORMATIONS))
        # Apply selected transformations
        for transform, transform_fn in zip(transformation_flags, TRANSFORMATIONS):
            if transform:
                sample = transform_fn(sample, encoder_decoder)

    return sample
