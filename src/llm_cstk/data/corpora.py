import re
from typing import Pattern


from .utils import *
from .utils import _ChatData, _RetrievalChatData


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
            SYSTEM: utt[SYSTEM],
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


SUPPORT_FEEDBACK: str = 'support_feedback'


class ChatDataS(_RetrievalChatData):
    DATA_SET_ID = 'chat_data_s'
    CUSTOM_COLS = [TITLE, SUPPORT_FEEDBACK]
    # Sys messages RegExes
    SYS_MSG_REGEX: List[Pattern[str]] = [
        re.compile(r"ALLEGARE i seguenti files :\s?\r\n(.*)"),
        re.compile(r"Stato ticket passa a CHIUSO\r\n\r\nVALUTAZIONE E NOTE FINALI TICKET \r\n\r\n\d\d? - (.*)"),
        re.compile(r"VALUTAZIONE E NOTE FINALI TICKET \r\n\r\n\d\d? - (.*)"),
        re.compile(r"VALUTAZIONE E NOTE FEEDBACK \r\n\r\n\d\d? - (.*)"),
        re.compile(r"ATTACH the following files :\s\r\n(.*)"),
        re.compile(r"State ticket changes to CLOSE\r\n\r\nEVALUATION AND NOTES FINAL TICKET \r\n\r\n\d\d? - (.*)"),
        re.compile(r"EVALUATION AND NOTES FINAL TICKET \r\n\r\n\d\d? - (.*)"),
        re.compile(r"EVALUATION AND NOTES FEEDBACK \r\n\r\n\d\d? - (.*)")
    ]
    SYS_MSG_MARKERS: List[str] = [  # NOTE order is important, this is used in an hardcoded function
        'SEGNALARE ULTERIORI CONTROLLI EFFETTUATI',
        'ALLEGARE i seguenti files',
        'Stato ticket passa a',
        'VALUTAZIONE E NOTE',
        'REPORT FURTHER CHECKS',
        'ATTACH the following files',
        'State ticket changes to',
        'EVALUATION AND NOTES'
    ]
    # Data types
    D_TYPES = {
        "ID_TICKET": int,
        "DESCRIZIONE_ITEM": str,
        "ID_CHAT": int,
        "ChatSenderUserId": str,
        "TESTO": str,
        "STATUS": int,
        "ID_MODELLO": int,
        "ID_CARROZZERIA": int,
        "MY": str,
        "ID_ASPIRAZIONE": int,
        "ID_CILINDRATA": int,
        "ID_DISTRIBUZIONE": int,
        "ID_TRAS_CAMBIO": int,
        "ID_ALIMENTAZIONE": int,
        "ID_TRAZIONE": int,
        "ID_FRENI": int,
        "ID_SERVOSTERZO": int,
        "ID_AC": int,
        "ID_AUDIO": int,
        "ID_AVVIO": int,
        "ID_START_STOP": int,
        "ID_SISTEMA": int,
        "VAL_SINTOMI": int,
        "KM": float,
        "ID_CVett": int,
        "ID_CClim": int,
        "ChkFVC": str,
        "DTC1": str,
        "DTC2": str,
        "DTC3": str,
        "DTC4": str,
        "DTC5": str,
        "DTC6": str,
        "VAL_FEEDBACK": 'Int64',
        "NOTE_FEEDBACK": 'Int64',
        "TicketCreateUserId": str,
        "TicketUpdateUserId": str,
        "LIVELLO_UPDATE": str,
        "DATE_TIME_UPDATE": str,
        "ID_FRENO_STAZ": int,
        "NumeroTicket": str,
        "InCaricoA": str,
        "Tempo1": 'Int64',
        "Tempo2": 'Int64',
        "Tempo3": 'Int64'
    }
    #
    NA_FILL_VALUE: int = -1
    UTTERANCE_COLS: List[str] = ['TESTO', 'ID_CHAT', 'ChatSenderUserId']
    VEHICLE_COLS_MAPPINGS: Dict[str, str] = {
        'ID_MODELLO': 'Codice modello',
        'ID_CARROZZERIA': 'Codice carrozzeria',
        'MY': 'Anno produzione',
        'KM': 'Km percorsi'
    }
    VEHICLE_COLS_MAPPINGS_EXTENDED: Dict[str, str] = {
        'ID_MODELLO': 'Codice modello',
        'ID_CARROZZERIA': 'Codice carrozzeria',
        'MY': 'Anno produzione',
        'KM': 'Km percorsi',
        'ID_ASPIRAZIONE': 'Codice aspirazione',
        'ID_CILINDRATA': 'Codice cilindrata',
        'ID_DISTRIBUZIONE': 'Codice distribuzione',
        'ID_TRAS_CAMBIO': 'Codice trasmissione cambio',
        'ID_ALIMENTAZIONE': 'Codice alimentazione',
        'ID_TRAZIONE': 'Codice trazione',
        'ID_FRENI': 'Codice freni',
        'ID_SERVOSTERZO': 'Codice servosterzo',
        'ID_AC': 'Codice impianto AC',
        'ID_AUDIO': 'Codice audio',
        'ID_AVVIO': 'Codice avvio',
        'ID_START_STOP': 'Codice start-stop',
        'ID_SISTEMA': 'Codice sistema',
        'ID_FRENO_STAZ': 'Codice freno di stazionamento idraulico',
        'VAL_SINTOMI': 'Codice sintomi'
    }
    DATE_DEFAULT: str = 'unknown'
    ABSTRACT_DEFAULT: str = 'Not available'
    SPEAKER_DEFAULT: str = 'Unknown'
    TEXT_DEFAULT: str = '###'
    # Column keys
    DIALOGUE_ID: str = 'ID_TICKET'
    DATE = 'DATE_TIME_UPDATE'
    UTTERANCE_ID: str = 'ID_CHAT'
    ABSTRACT: str = 'DESCRIZIONE_ITEM'
    SPEAKER: str = 'ChatSenderUserId'
    TEXT: str = 'TESTO'
    VAL_FEEDBACK: str = 'VAL_FEEDBACK'

    def __init__(
            self, *args, holdout: Optional[Union[float, int]] = 0.2, use_extended_metadata: bool = False, **kwargs
    ):
        self.use_extended_metadata: bool = use_extended_metadata
        #
        super(ChatDataS, self).__init__(*args, holdout=holdout, **kwargs)

    @staticmethod
    def _preprocess_text(text: str) -> str:
        for u_code_sym, replace_sym in UNICODE_SWITCH_LIST:
            text = text.replace(u_code_sym, replace_sym)  # Taken from ParlAI preprocessing
        # text = re.sub(r'\.(\w)', r' . \1', text)  # Taken from ParlAI preprocessing
        # text = re.sub('[ \t\n]+', ' ', text)
        text = text.replace('\r', '')
        text = text.strip()

        return text

    def _check_sys_msg(self, txt: str) -> bool:
        txt = txt.strip()
        # if any(regex.match(txt) for regex in self.SYS_MSG_REGEX):
        if any(marker in txt for marker in self.SYS_MSG_MARKERS):
            return True
        else:
            return False

    def _preprocess_utterance(self, utt: pd.Series, *args, **kwargs) -> Dict[str, Union[str, bool]]:
        # TODO add flag for system messages
        utterance: Dict[str, str] = {
            SPEAKER: utt[self.SPEAKER] if utt[self.SPEAKER] != self.NA_FILL_VALUE else self.SPEAKER_DEFAULT,
            SYSTEM: self._check_sys_msg(utt[self.TEXT] if utt[self.TEXT] != self.NA_FILL_VALUE else self.TEXT_DEFAULT),
            TEXT: self._preprocess_text(utt[self.TEXT] if utt[self.TEXT] != self.NA_FILL_VALUE else self.TEXT_DEFAULT)
        }

        return utterance

    def _preprocess_metadata(self, meta: pd.Series, *args, **kwargs) -> str:
        # Extract values
        # TODO add other structured info
        ticket_id = meta[self.DIALOGUE_ID] if meta[self.DIALOGUE_ID] != self.NA_FILL_VALUE else None
        date = meta[self.DATE] if meta[self.DATE] != self.NA_FILL_VALUE else self.DATE_DEFAULT
        vehicle = '\n'.join(
            f'{v}: {meta[k]}'
            for k, v in (
                 self.VEHICLE_COLS_MAPPINGS_EXTENDED if self.use_extended_metadata else self.VEHICLE_COLS_MAPPINGS
            ).items()
        )
        abstract = meta[self.ABSTRACT] if meta[self.ABSTRACT] != self.NA_FILL_VALUE else self.ABSTRACT_DEFAULT
        # Prepare metadata string
        metadata = f'Ticket: {ticket_id} -- Date: {date}\n\n' \
                   f'Veicolo:\n{vehicle}\n\n' \
                   f'Item description:\n{abstract}'

        return metadata

    def _preprocess_dialogue(
            self, df: pd.DataFrame, *args, **kwargs
    ) -> Dict[str, Union[str, Dict[str, Union[str, bool]]]]:
        # Prepare data
        df = df.sort_values(self.UTTERANCE_ID)
        metadata: pd.Series = df.iloc[0].drop(self.UTTERANCE_COLS)
        # Compose dialogue dict
        dialogue: Dict[str, Union[str, Dict[str, str]]] = {
            SPLIT: self.split,
            DATA_SET_ID: self.DATA_SET_ID,
            DIALOGUE_ID: metadata[self.DIALOGUE_ID],
            INFO: self._preprocess_metadata(metadata),
            UTTERANCES: [
                self._preprocess_utterance(utterance) for _, utterance in df.iterrows()
            ],
            TITLE: self._preprocess_text(
                metadata[self.ABSTRACT] if metadata[self.ABSTRACT] != self.NA_FILL_VALUE else self.TEXT_DEFAULT
            ),
            SUPPORT_FEEDBACK: metadata[self.VAL_FEEDBACK] if metadata[self.VAL_FEEDBACK] else self.TEXT_DEFAULT
        }

        return dialogue

    def _load_data_frame(self) -> pd.DataFrame:
        # Load data frame
        df = pd.read_csv(self.path, dtype=self.D_TYPES)
        # TODO preprocess NaN values in a better way
        df = df.fillna(value=self.NA_FILL_VALUE)
        df[self.DATE] = pd.to_datetime(df[self.DATE], format='%Y%m%d%H%M%S')

        return df

    def _load_samples(self) -> List[Dict[str, Union[str, Dict[str, Union[str, bool]]]]]:
        # Load data frame
        df: pd.DataFrame = self._load_data_frame()
        # Get dialogue IDs
        dialogue_ids = df[self.DIALOGUE_ID].unique()
        # Select current split indices
        idxs: List[int] = self._get_split_indices(len(dialogue_ids))
        idxs = self._get_sample_indices(idxs)
        # Filter dialogue IDs
        dialogue_ids = dialogue_ids[idxs]
        df = df[df[self.DIALOGUE_ID].isin(dialogue_ids)]
        # Preprocess dialogues
        with parallel_backend(self.JOBLIB_BACKEND, n_jobs=self.N_JOBS):
            data: List[Dict[str, Union[str, Dict[str, Union[str, bool]]]]] = Parallel(verbose=self.VERBOSITY_LEVEL)(
                delayed(self._preprocess_dialogue)(dialogue_df)
                for _, dialogue_df in df.groupby(by=self.DIALOGUE_ID, sort=False)
            )

        return data

    def drop_sys_messages(self):
        for sample in self.data:
            sample[UTTERANCES] = [utterance for utterance in sample[UTTERANCES] if not utterance[SYSTEM]]


class ChatDataN(_RetrievalChatData):
    DATA_SET_ID = 'chat_data_n'
    ...


class ChatDataL(_RetrievalChatData):
    DATA_SET_ID = 'chat_data_l'
    ...
