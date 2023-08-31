from .utils import *
from .utils import _ChatData


class ChatData(_ChatData):
    UTTERANCE_COLS: List[str] = [SPEAKER, TEXT]

    def __init__(self, path: str, data_set_id: Optional[str] = None, split:  Optional[Split] = None, **kwargs):
        # Parse file path
        tmp_data_set_id, tmp_split = self._parse_file_path(path)
        #
        self.data_set_id: str = data_set_id if data_set_id is not None else tmp_data_set_id
        split = split if split is not None else tmp_split
        #
        super(ChatData, self).__init__(path, split=split, **kwargs)

    @staticmethod
    def _parse_file_path(path) -> Tuple[str, Optional[Split]]:
        # Get file name
        file_name = os.path.basename(path)
        # Remove file extension
        file_name, _ = os.path.splitext(file_name)
        #
        corpus_id: str = file_name
        split: Optional[Split] = None
        if '_' in file_name:
            *tmp, split = file_name.split('_')
            if split in SPLITS:
                corpus_id = '_'.join(tmp)
            else:
                split = None

        return corpus_id, split

    def _preprocess_utterance(self, utt: pd.Series, *args, **kwargs) -> Dict[str, Union[str, bool]]:
        utterance: Dict[str, Union[str, bool]] = {
            SPEAKER: utt[SPEAKER],
            SYSTEM_FLAG: utt[SYSTEM_FLAG],
            TEXT: self._preprocess_text(utt[TEXT]),
        }

        return utterance

    def _preprocess_metadata(self, meta: pd.Series, *args, **kwargs) -> Optional[str]:
        metadata: str = meta[INFO]

        return metadata

    def _preprocess_dialogue(
            self, df: pd.DataFrame, *args, **kwargs
    ) -> Dict[str, Union[str, Dict[str, Union[str, bool]]]]:
        # Prepare data
        df = df.sort_values(UTTERANCE_IDX)
        metadata: pd.Series = df.iloc[0].drop(self.UTTERANCE_COLS)
        # Compose dialogue dict
        dialogue: Dict[str, Union[str, Dict[str, Optional[str]]]] = {
            SPLIT: self.split,
            DATA_SET_ID: self.data_set_id,
            DIALOGUE_ID: metadata[DIALOGUE_ID],
            INFO: self._preprocess_metadata(metadata),
            UTTERANCES: [
                self._preprocess_utterance(utterance) for _, utterance in df.iterrows()
            ]
        }

        return dialogue

    def _load_samples(self) -> List[Dict[str, Union[str, Dict[str, Union[str, bool]]]]]:
        # Load data frame
        df = pd.read_csv(self.path)
        df = df.fillna(UNK)
        # Preprocess samples
        data: List[Dict[str, Union[str, Dict[str, str]]]] = [
            self._preprocess_dialogue(dialogue_df) for _, dialogue_df in df.groupby(by=DIALOGUE_ID, sort=False)
        ]

        return data
