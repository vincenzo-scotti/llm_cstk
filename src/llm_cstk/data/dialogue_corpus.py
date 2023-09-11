import bz2
import pickle
import copy
import yaml

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoConfig
from transformers import BatchEncoding

from .corpora import ChatData
from .preparation import prepare_sample, sample_to_string, _check_fit_in_token_window
from .utils import *


"""
    Expected sample format:
    
    sample = {
        'split': ...
        'data_set_id': ...
        'dialogue_id': ...
        'metadata': ...
        'context': [
            {'speaker': ..., sys: ..., 'text': ...},
            {'speaker': ..., sys: ..., 'text': ...},
            {'speaker': ..., sys: ..., 'text': ...},
            ...
        ]
        'response': {'speaker': ..., sys: ..., 'text': ...},
    }
    
    or 
    
    sample = {
        'split': ...
        'data_set_id': ...
        'dialogue_id': ...
        'metadata': ...
        'utterances': [
            {'speaker': ..., sys: ..., 'text': ...},
            {'speaker': ..., sys: ..., 'text': ...},
            {'speaker': ..., sys: ..., 'text': ...},
            ...
        ]
    }
    
"""


class ChatDataset(Dataset):
    _PARAMS: List[str] = ['corpus_list', 'stride', 'encoder_decoder']

    def __init__(
            self,
            split: Split,
            transformer: str,
            cache_dir_path: str,
            corpora_dir_path: str,
            corpus_list: List[str],
            corpus_prefix: str = 'chat_data',
            stride: Optional[int] = None,
            tokenisation: Optional[Dict] = None
    ):
        super(ChatDataset, self).__init__()
        # Instance parameters
        self.corpora_dir_path = corpora_dir_path
        self.corpus_list = corpus_list
        self.stride = stride
        # Loader parameters
        self.split: Split = split
        self.transformer: str = transformer
        tokenisation = tokenisation if tokenisation is not None else dict()
        self._tokeniser: PreTrainedTokenizer = AutoTokenizer.from_pretrained(transformer, **tokenisation)
        self.encoder_decoder = AutoConfig.from_pretrained(self.transformer).is_encoder_decoder
        self.corpus_cache_file_path: Optional[str] = None
        if cache_dir_path is not None:
            if not os.path.exists(cache_dir_path):
                os.mkdir(cache_dir_path)
            self.corpus_cache_file_path = os.path.join(
                cache_dir_path, f'{corpus_prefix}_{split}_{self._get_cache_id()}.{CACHE_COMPRESSED_EXT}'
            )
        # Load data cache if available else prepare data from scratch
        self.data: List[Dict]
        self._load_data()
        self.metadata: Dict[str, Optional[Dict]]
        self._load_metadata()

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data)

    def __getitem__(self, index: int):
        # Get utterances from data set
        sample = self.data[index]
        metadata = self.metadata.get(sample[DATA_SET_ID])
        if metadata is not None:
            sample = copy.deepcopy(sample)
            sample.update(metadata)

        return sample

    def _get_cache_id(self) -> str:
        # TODO avoid using numpy for unsigned int casting
        cache_id: str = str(np.uint(hash('_'.join(f'{getattr(self, param)}' for param in self._PARAMS))))

        return cache_id

    def _serialise_data_cache(self):
        if self.corpus_cache_file_path is not None:
            # Save compressed pickle file
            with bz2.BZ2File(self.corpus_cache_file_path, 'wb') as f:
                pickle.dump(self.data, f)

    def _load_data_cache(self):
        if self.corpus_cache_file_path is not None:
            # Load compressed pickle file
            with bz2.BZ2File(self.corpus_cache_file_path, 'rb') as f:
                self.data = pickle.load(f)

    def _chunk_dialogue(self, sample: Dict) -> List[Dict]:
        # Check whether dialogue fits in token window
        sample_str = sample_to_string(sample, False, self._tokeniser, self.encoder_decoder)
        if _check_fit_in_token_window(sample_str, self._tokeniser, self.encoder_decoder):
            return [sample]
        else:
            # Find minimum dialogue fitting in window:
            tmp_sample = copy.deepcopy(sample)
            tmp_sample_str = sample_to_string(tmp_sample, False, self._tokeniser, self.encoder_decoder)
            while not _check_fit_in_token_window(tmp_sample_str, self._tokeniser, self.encoder_decoder):
                if len(tmp_sample[UTTERANCES]) > 0:
                    tmp_sample[UTTERANCES].pop()
                tmp_sample_str = sample_to_string(tmp_sample, False, self._tokeniser, self.encoder_decoder)
            # Iteratively build new dialogues sliding forward the end of the sequence
            samples = list()
            for i in range(len(tmp_sample[UTTERANCES]), len(sample[UTTERANCES]) + 1, self.stride):
                tmp_sample = copy.deepcopy(sample)
                tmp_sample[UTTERANCES] = tmp_sample[UTTERANCES][:i]
                samples.append(tmp_sample)

            return samples

    def _load_data(self):
        # Check whether data cache already exists
        if os.path.exists(self.corpus_cache_file_path):
            # Load data from cache
            self._load_data_cache()
        else:
            # Load data from preprocessed corpora
            self.data = list()
            # Iterate over selecte corpora
            for corpus in self.corpus_list:
                chat_data: ChatData = ChatData(
                    os.path.join(self.corpora_dir_path, f'{corpus}_{self.split}.{PREPROCESSED_DATA_EXT}'),
                    data_set_id=corpus,
                    split=self.split
                )
                if self.encoder_decoder:
                    self.data += chat_data.get_utterances()
                else:
                    self.data += sum(
                        [self._chunk_dialogue(sample) for sample in chat_data.get_dialogues()], start=list()
                    )
            # Save data cache
            self._serialise_data_cache()

    def _load_metadata(self):
        self.metadata = dict()
        # Iterate over corpora
        for corpus in self.corpus_list:
            metadata_file_path = os.path.join(
                self.corpora_dir_path, f'{corpus}_{METADATA_FILE_SUFFIX}.{METADATA_FILE_EXT}'
            )
            metadata: Optional[Dict] = None
            if os.path.exists(metadata_file_path):
                with open(metadata_file_path) as f:
                    metadata = yaml.full_load(f)[CHAT][TRANSDUCER if self.encoder_decoder else CAUSAL]
            self.metadata[corpus] = metadata

    def _prepare_sample(self, sample: Dict) -> Union[str, Tuple[str, str]]:
        # Use utility function to prepare sample
        sample_str: Union[str, Tuple[str, str]] = prepare_sample(
            sample,
            self._tokeniser,
            # augmentation=self.split == TRAIN,  # TODO restore this version
            augmentation=(sample[SPLIT] if SPLIT in sample else self.split) == TRAIN,
            encoder_decoder=self.encoder_decoder
        )

        return sample_str

    def collate(
            self, samples: List[Dict]
    ) -> Tuple[
        Union[Tuple[BatchEncoding, torch.tensor], Tuple[BatchEncoding, BatchEncoding, torch.tensor]], List[Dict], Split
    ]:
        tensor_data: Union[Tuple[BatchEncoding, torch.tensor], Tuple[BatchEncoding, BatchEncoding, torch.tensor]]
        # Depending on the selected model and _tokeniser prepare input tensors
        if self.encoder_decoder:
            src_strings, tgt_strings = [*zip(*[self._prepare_sample(sample) for sample in samples])]
            tgt_strings = [self._tokeniser.pad_token + tgt_str for tgt_str in tgt_strings]
            src_encoding: BatchEncoding = self._tokeniser(src_strings, return_tensors='pt', padding=True)
            tgt_encoding: BatchEncoding = self._tokeniser(
                tgt_strings, return_tensors='pt', padding=True, truncation=True
            )
            mask = ~tgt_encoding.attention_mask.bool()
            labels: torch.tensor = tgt_encoding.input_ids.clone()
            labels[mask] = IGNORE_IDX
            tensor_data = (src_encoding, tgt_encoding, labels)
        else:
            input_strings = [
                (self._tokeniser.bos_token if self._tokeniser.bos_token is not None else self._tokeniser.eos_token) +
                self._prepare_sample(sample) + self._tokeniser.eos_token
                for sample in samples
            ]
            input_encodings: BatchEncoding = self._tokeniser(
                input_strings, return_tensors='pt', padding=True, truncation=True
            )
            mask = ~input_encodings.attention_mask.bool()
            labels: torch.tensor = input_encodings.input_ids.clone()
            labels[mask] = IGNORE_IDX
            tensor_data = (input_encodings, labels)
        # Compose mini-batch
        mini_batch: Tuple[
            Union[Tuple[BatchEncoding, torch.tensor], Tuple[BatchEncoding, BatchEncoding, torch.tensor]],
            List[Dict],
            Split
        ] = (tensor_data, samples, self.split)

        return mini_batch

    def huggingface_collate(self, samples: List[Dict]) -> Dict[str, torch.tensor]:
        if self.encoder_decoder:
            src_strings, tgt_strings = [*zip(*[self._prepare_sample(sample) for sample in samples])]
            tgt_strings = [self._tokeniser.pad_token + tgt_str for tgt_str in tgt_strings]
            src_encoding: BatchEncoding = self._tokeniser(src_strings, return_tensors='pt', padding=True)
            tgt_encoding: BatchEncoding = self._tokeniser(
                tgt_strings, return_tensors='pt', padding=True, truncation=True
            )
            mask = ~tgt_encoding.attention_mask.bool()
            labels: torch.tensor = tgt_encoding.input_ids.clone()
            labels[mask] = -100

            return {
                'input_ids': src_encoding.input_ids,
                'attention_mask': src_encoding.attention_mask,
                'decoder_input_ids': tgt_encoding.input_ids,
                'decoder_attention_mask': tgt_encoding.attention_mask,
                'labels': labels
            }
        else:
            input_strings = [
                (self._tokeniser.bos_token if self._tokeniser.bos_token is not None else self._tokeniser.eos_token) +
                self._prepare_sample(sample) + self._tokeniser.eos_token
                for sample in samples
            ]
            input_encodings: BatchEncoding = self._tokeniser(
                input_strings, return_tensors='pt', padding=True, truncation=True
            )
            mask = ~input_encodings.attention_mask.bool()
            labels: torch.tensor = input_encodings.input_ids.clone()
            labels[mask] = -100

            return {
                'input_ids': input_encodings.input_ids,
                'attention_mask': input_encodings.attention_mask,
                'labels': labels
            }

    def as_strings(self) -> List[str]:
        if self.encoder_decoder:
            raise NotImplementedError()
        else:
            return [self._prepare_sample(sample) + self._tokeniser.eos_token for sample in self.data]
