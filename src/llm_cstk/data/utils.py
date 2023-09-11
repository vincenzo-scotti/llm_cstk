import os

import math
import pandas as pd

from sklearn.model_selection import train_test_split
import random

from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend

from llm_cstk.retrieval.pyterrier_extensions.utils import *
from llm_cstk.utils.common import *

# Types

Split: TypeAlias = Literal['train', 'validation', 'test', 'dev']

# Constants

TRAIN: str = 'train'
VALIDATION: str = 'validation'
TEST: str = 'test'
DEV: str = 'dev'

SPLITS: List[str] = [TRAIN, VALIDATION, TEST, DEV]

TRANSDUCER: str = 'transducer'
CAUSAL: str = 'causal'

ARCHITECTURES: List[str] = [TRANSDUCER, CAUSAL]

IGNORE_IDX: int = -1

SPLIT: str = 'split'
DATA_SET_ID: str = 'data_set_id'
DIALOGUE_ID: str = 'dialogue_id'
INFO: str = 'info'
UTTERANCES: str = 'utterances'

INSTRUCTIONS: str = 'instructions'
INSTRUCTIONS_ALTERNATIVES: str = 'instructions_alternatives'

UTTERANCE_IDX: str = 'utterance_idx'

SPEAKER: str = 'speaker'
SYSTEM_FLAG: str = 'sys'
TEXT: str = 'text'

CONTEXT: str = 'context'
RESPONSE: str = 'response'

PREPROCESSED_DATA_EXT: str = 'csv'
CACHE_COMPRESSED_EXT: str = 'pbz2'
METADATA_FILE_SUFFIX: str = 'metadata'
METADATA_FILE_EXT: str = 'yml'

# Sample structure

SEP: str = '\n'
BLOCK_SEP: str = '\n\n'
ELLIPS: str = '(...)'
SPEAKER_SEP: str = ': '
UTTERANCES_SEP: str = '\n---\n'
UNK: str = '###'

INFO_ID: str = 'Info:'
CONTEXT_ID: str = 'Dialogue:'
RESPONSE_ID: str = 'Response:'
SYSTEM_ID: str = 'System'
REPORT_ID: str = 'Report'

# Tasks

CHAT: str = 'chat'

# Substitutes for non-unicode special characters
UNICODE_SWITCH_LIST: List[Tuple[str, str]] = [
    ("\u2019", "'"),
    ("\u2018", "'"),
    ("\u201d", '"'),
    ("\u201c", '"'),
    ("\u2014", "--"),
    ("\u2013", "--"),
    ("\u3002", ". "),
    ("\u2032", "'"),
    ("\u3001", ", ")
]


# Abstract classes and interfaces

class _ChatData:
    DATA_SET_ID: Optional[str] = None
    CUSTOM_COLS: Optional[List[str]] = None
    # Joblib params
    JOBLIB_BACKEND: str = 'threading'
    N_JOBS: int = -1
    VERBOSITY_LEVEL: int = 2

    def __init__(
            self,
            path: str,
            split: Optional[Split] = None,
            holdout: Optional[Union[float, int]] = None,
            sample: Optional[Union[float, int]] = None,
            random_seed: Optional[int] = None,
    ):
        self.path: str = path
        self.split: Optional[Split] = split
        self.holdout: Optional[Union[float, int]] = holdout
        self.sample: Optional[Union[float, int]] = sample
        self.random_seed: Optional[int] = random_seed
        self.data: Optional[List[Dict[str, Union[str, List[Dict[str, str]]]]]] = self._load_samples()

    @staticmethod
    def _preprocess_text(text: str) -> str:
        for u_code_sym, replace_sym in UNICODE_SWITCH_LIST:
            text = text.replace(u_code_sym, replace_sym)  # Taken from ParlAI preprocessing
        # text = re.sub(r'\.(\w)', r' . \1', text)  # Taken from ParlAI preprocessing
        # text = re.sub('[ \t\n]+', ' ', text)
        text = text.strip()

        return text

    def _preprocess_utterance(self, *args, **kwargs) -> Dict[str, str]:
        raise NotImplementedError()

    def _preprocess_metadata(self, *args, **kwargs) -> str:
        raise NotImplementedError()

    def _preprocess_dialogue(self, *args, **kwargs) -> Dict[str, Union[str, Dict[str, Union[str, bool]]]]:
        raise NotImplementedError()

    def _get_sample_indices(self, idxs: List[int]):
        # Apply subsampling if required
        if self.sample is not None:
            # Get number of samples to collect
            if isinstance(self.sample, int):
                n_samples = self.sample
            else:
                n_samples = int(math.ceil(self.sample * len(idxs)))
            # Subsample data set unless the number of samples to take is equal to the number of samples available
            if n_samples != len(idxs):
                idxs = random.sample(idxs, n_samples)

        return idxs

    def _get_split_indices(self, n: int) -> List[int]:
        # Get indices list
        idxs = list(range(n))
        # Apply train-validation-test split if required
        if self.holdout is not None and self.split is not None:
            # Do train-validation-test split on the indices
            train_idxs, test_idxs = train_test_split(idxs, test_size=self.holdout, random_state=self.random_seed)
            train_idxs, val_idxs = train_test_split(train_idxs, test_size=self.holdout, random_state=self.random_seed)
            # Get list of current split indices
            if self.split == 'train':
                idxs = train_idxs
            elif self.split == 'validation':
                idxs = val_idxs
            elif self.split == 'test':
                idxs = test_idxs
            else:
                raise ValueError(f'Unknown value for data set split: {self.split}')

        return idxs

    def _load_samples(self) -> List[Dict[str, Union[str, Dict[str, Union[str, bool]]]]]:
        raise NotImplementedError()

    def get_dialogues(self) -> List[Dict[str, Union[str, List[Dict[str, Union[str, bool]]]]]]:
        return self.data

    def get_utterances(self) -> List[Dict[str, Union[str, List[Dict[str, Union[str, bool]]]]]]:
        # Ravel all dialogues into context-response pairs
        data: List[Dict[str, Union[str, List[Dict[str, str]]]]] = [
            {
                SPLIT: self.split,
                DATA_SET_ID: getattr(self, 'data_set_id', self.DATA_SET_ID),
                DIALOGUE_ID: dialogue[DIALOGUE_ID],
                INFO: dialogue[INFO],
                CONTEXT: dialogue[UTTERANCES][:utterance_idx],
                RESPONSE: utterance
            }
            for dialogue in self.data for utterance_idx, utterance in enumerate(dialogue[UTTERANCES])
        ]

        return data

    def _get_flattened_data(self) -> List[Dict]:
        return [
            {
                SPLIT: self.split,
                DATA_SET_ID: self.DATA_SET_ID,
                DIALOGUE_ID: dialogue[DIALOGUE_ID],
                INFO: dialogue[INFO],
                UTTERANCE_IDX: idx,
                SPEAKER: utterance[SPEAKER],
                SYSTEM_FLAG: utterance[SYSTEM_FLAG],
                TEXT: utterance[TEXT]
            } | {
                key: dialogue[key] if key in dialogue else utterance[key]
                for key in self.CUSTOM_COLS
            }
            for dialogue in self.data for idx, utterance in enumerate(dialogue[UTTERANCES])
        ]

    def serialise_data(self, dir_path: str, file_name: Optional[str] = None):
        # Set file name
        # TODO manage subsampling
        if file_name is None:
            file_name = f'{self.DATA_SET_ID}_{self.split}.csv' if self.split is not None else f'{self.DATA_SET_ID}.csv'
        # File path
        path: str = os.path.join(dir_path, file_name)
        #
        data = self._get_flattened_data()
        df = pd.DataFrame(data)
        # Serialise CSV at target path
        df.to_csv(path, index=False)


class _RetrievalChatData(_ChatData):
    CUSTOM_COLS: Optional[List[str]] = [TITLE]

    @staticmethod
    def _get_dialogue_string(utterances: List[Dict]) -> str:
        return '\n\n'.join(f'- {utterance[SPEAKER]}: {utterance[TEXT]}' for utterance in utterances)

    def _sample_to_doc(self, sample: Dict, idx: int) -> Dict:
        #
        title: str = sample[TITLE]
        body: str = f'{sample[INFO]}\n\n{self._get_dialogue_string(sample[UTTERANCES])}'
        #
        doc: Dict = {
            DOCID: idx,
            DOCNO: sample[DIALOGUE_ID],
            TEXT: f'{title}\n\n{body}',
            TITLE: title,
            BODY: body,
        }

        return doc

    def to_doc_collection(self, path: Optional[str] = None) -> List[Dict]:
        with parallel_backend(self.JOBLIB_BACKEND, n_jobs=self.N_JOBS):
            docs: List[Dict] = Parallel(verbose=self.VERBOSITY_LEVEL)(
                delayed(self._sample_to_doc)(sample, idx) for idx, sample in enumerate(self.data)
            )

        if path is not None:
            pd.DataFrame(docs).to_csv(path)

        return docs

    def _sample_to_doc_chunk(self, sample: Dict, doc_idx: int, chunk_idx: int, s_idx: int, e_idx: int) -> Dict:
        #
        title: str = sample[TITLE]
        body: str = self._get_dialogue_string(sample[UTTERANCES][s_idx:e_idx])
        if s_idx > 0:
            body = f'(...)\n\n{body}'
        if e_idx < len(sample[UTTERANCES]):
            body = f'{body}\n\n(...)'
        #
        doc: Dict = {
            DOCID: doc_idx,  # TODO check if this is correct
            DOCNO: f'{sample[DIALOGUE_ID]}%p{chunk_idx}',
            TEXT: f'{title}\n\n{body}',
            TITLE: title,
            BODY: body,
        }

        return doc

    def _sample_info_to_doc_chunk(self, sample: Dict, doc_idx: int) -> Dict:
        #
        title: str = sample[TITLE]
        body: str = sample[INFO]
        #
        doc: Dict = {
            DOCID: doc_idx,  # TODO check if this is correct
            DOCNO: f'{sample[DIALOGUE_ID]}%p0',
            TEXT: f'{title}\n\n{body}',
            TITLE: title,
            BODY: body,
        }

        return doc

    def to_doc_chunk_collection(self, size: int, stride: int, path: Optional[str] = None) -> List[Dict]:
        with parallel_backend(self.JOBLIB_BACKEND, n_jobs=self.N_JOBS):
            docs: List[Dict] = Parallel(verbose=self.VERBOSITY_LEVEL)(
                delayed(self._sample_info_to_doc_chunk)(sample, d_idx)
                for d_idx, sample in enumerate(self.data)
            ) + Parallel(verbose=self.VERBOSITY_LEVEL)(
                delayed(self._sample_to_doc_chunk)(
                    sample, d_idx, c_idx, s_idx, max(s_idx + size, len(sample[UTTERANCES]))
                )
                for d_idx, sample in enumerate(self.data)
                for c_idx, s_idx in enumerate(range(0, len(sample[UTTERANCES]), stride), start=1)
            )

        if path is not None:
            pd.DataFrame(docs).to_csv(path)

        return docs

