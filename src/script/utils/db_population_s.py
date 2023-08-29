# Populate DB
import sys

from typing import Dict
from argparse import ArgumentParser, Namespace

import pandas as pd

from tqdm import tqdm

from search.models import ChatModel, ChatTurnModel


D_TYPES: Dict = {
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
SYS_MSG = [  # NOTE order is important, this is used in an hardcoded function
    'SEGNALARE ULTERIORI CONTROLLI EFFETTUATI',
    'ALLEGARE i seguenti files',
    'Stato ticket passa a',
    'VALUTAZIONE E NOTE',
    'REPORT FURTHER CHECKS',
    'ATTACH the following files',
    'State ticket changes to',
    'EVALUATION AND NOTES'
]


def main(args: Namespace):
    # Data
    data = pd.read_csv(args.data_file_path)
    data = data.fillna(value=-1)
    data['DATE_TIME_UPDATE'] = pd.to_datetime(data['DATE_TIME_UPDATE'], format='%Y%m%d%H%M%S')
    # Group data
    grouped_data = data.groupby('ID_TICKET', sort=False)
    grouped_data = grouped_data.first().reset_index().drop(['TESTO', 'ID_CHAT', 'ChatSenderUserId'], axis=1)
    # Chats
    chat_mapping = dict()
    for idx, row in grouped_data.iterrows():
        record = ChatModel(
            chat_id=row['ID_TICKET'],
            item_description=row['DESCRIZIONE_ITEM'] if row['DESCRIZIONE_ITEM'] != -1 else None,
            date=row['DATE_TIME_UPDATE']
        )
        chat_mapping[row['ID_TICKET']] = record
        record.save()
    # Messages
    for idx, row in data.iterrows():
        record = ChatTurnModel(
            message_id=row['ID_CHAT'],
            chat_id=chat_mapping[row['ID_TICKET']],
            speaker=row['ChatSenderUserId'],
            text=row['TESTO'],
            automatic=any(sm in row['TESTO'] for sm in SYS_MSG)
        )
        record.save()

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser()
    # Add arguments to parser
    args_parser.add_argument(
        '--data_file_path',
        type=str,
        help="Path to the CSV file containing the data samples to add to the DB."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
