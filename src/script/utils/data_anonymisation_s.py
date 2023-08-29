import os
import sys
import logging
from datetime import datetime

from typing import Dict, Optional, Tuple, List
from argparse import ArgumentParser, Namespace

import yaml
import json
import html

import pandas as pd

import re
import stanza
import spacy

from tqdm import tqdm

from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend


REPL_STR: str = '<|null|>'
SYS_MSG: List[str] = [  # NOTE order is important, this is used in an hardcoded function
    'SEGNALARE ULTERIORI CONTROLLI EFFETTUATI',
    'ALLEGARE i seguenti files',
    'Stato ticket passa a',
    'VALUTAZIONE E NOTE',
    'REPORT FURTHER CHECKS',
    'ATTACH the following files',
    'State ticket changes to',
    'EVALUATION AND NOTES'
]


def clean_text(doc: str) -> str:
    # These passages are hard-coded
    # Check for html coder
    doc = html.unescape(doc)
    # Check for ill-formed system messages
    fixed = False
    i = 0
    while not fixed and i < len(SYS_MSG):
        s_idx = doc.find(SYS_MSG[i])
        if s_idx > 0:
            doc = doc[s_idx:]
            fixed = True
        i += 1

    return doc


def extract_entity_info(doc: str, configs: Dict) -> Tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, int]]]:
    # Extract stats and info
    entities_data: Dict[str, Dict[str, List[str]]] = {'spacy': dict(), 'stanza': dict(), 'regex': dict(), 'misc': dict()}
    entities_stats: Dict[str, Dict[str, int]] = {'spacy': dict(), 'stanza': dict(), 'regex': dict()}
    # Clean text
    doc = clean_text(doc)
    # Check if it is a system message
    if 'sys_msg' in configs:
        if any(sys_msg_regex.match(doc) for sys_msg_regex in configs['sys_msg']['regex']):
            # Check if it is a mixed system/user message
            for sys_msg_regex in configs['sys_msg']['anonymisation']:
                match = sys_msg_regex.findall(doc)
                if len(match) > 0:
                    match_string, *_ = match

                    return extract_entity_info(match_string, configs)
            else:
                return entities_data, entities_stats
    # Parse doc with external models if required
    spacy_doc: Optional = configs['spacy']['model'](doc) if 'spacy' in configs else None
    stanza_doc: Optional = configs['stanza']['model'](doc) if 'stanza' in configs else None
    # Iterate over entities to anonymise
    for entity_type, entity_options in configs['anonymisation'].items():
        if entity_type == 'misc':
            count = 0
            for text in entity_options:
                matches = re.findall(f'\\b{text}\\b', doc, flags=re.IGNORECASE)
                entities_data['misc'][text] = matches
                count += len(matches)
        else:
            if 'spacy' in entity_options and spacy_doc is not None:
                entities_data['spacy'][entity_type] = [
                    token.text for token in spacy_doc if token.ent_type_ in entity_options['spacy']
                ]
                entities_stats['spacy'][entity_type] = len(entities_data['spacy'][entity_type])
            if 'stanza' in entity_options:
                for sentence in stanza_doc.sentences:
                    entities_data['stanza'][entity_type] = [
                        entity.text for entity in sentence.ents if entity.type in entity_options['stanza']
                    ]
                    entities_stats['stanza'][entity_type] = len(entities_data['stanza'][entity_type])
            if 'regex' in entity_options:
                entities_data['regex'][entity_type] = [
                    match if isinstance(match, str) else match[0]
                    for match in entity_options['regex'].findall(doc)
                ]
                entities_stats['regex'][entity_type] = len(entities_data['regex'][entity_type])

    return entities_data, entities_stats


def extract_entity_info_sample(sample, configs):
    return {key: extract_entity_info(sample[key], configs) for key in configs['tgt_columns']}


def anonymise_string(doc: str, configs: Dict) -> str:
    # Clean text
    doc = clean_text(doc)
    # Check if it is a system message
    if 'sys_msg' in configs:
        if any(sys_msg_regex.match(doc) for sys_msg_regex in configs['sys_msg']['regex']):
            # Check if it is a mixed system/user message
            for sys_msg_regex in configs['sys_msg']['anonymisation']:
                match: re.Match = sys_msg_regex.search(doc)
                if match is not None:
                    s_idx, e_idx = match.regs[1]
                    doc = doc[:s_idx] + anonymise_string(doc[s_idx:e_idx], configs) + doc[e_idx:]

                    return doc
            else:
                return doc
    # Parse doc with external models if required
    spacy_doc: Optional = configs['spacy']['model'](doc) if 'spacy' in configs else None
    stanza_doc: Optional = configs['stanza']['model'](doc) if 'stanza' in configs else None
    # Iterate over entities to anonymise
    for entity_type, entity_options in configs['anonymisation'].items():
        if entity_type == 'misc':
            for text, repl in entity_options.items():
                doc = re.sub(f'\\b{text}\\b', repl, doc, flags=re.IGNORECASE)
        else:
            repl: str = entity_options.get('sub', REPL_STR)
            if 'spacy' in entity_options and spacy_doc is not None:
                for token in spacy_doc:
                    if token.ent_type_ in entity_options['spacy']:
                        doc = doc.replace(token.text, repl)
            if 'stanza' in entity_options:
                for sentence in stanza_doc.sentences:
                    for entity in sentence.ents:
                        if entity.type in entity_options['stanza']:
                            doc = doc.replace(entity.text, repl)
            if 'regex' in entity_options:
                doc = entity_options['regex'].sub(repl, doc)

    return doc


def anonymise_sample(sample, configs):
    return {
        key: anonymise_string(sample[key], configs) if key in configs['tgt_columns'] else sample[key]
        for key in configs.get('output_columns', sample.keys()) if key not in configs.get('drop_columns', set())
    }


def main(args: Namespace):
    # Init environment
    # Get date-time
    date_time: str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # Load configs
    with open(args.config_file_path) as f:
        configs: Dict = yaml.full_load(f)
    # Init logging
    logging.basicConfig(level=configs.get('log_level', 'INFO'))
    # Start Logging info
    logging.info("Script started and configuration file loaded")
    # Load models and compile regex expressions
    if 'sys_msg' in configs:
        configs['sys_msg']['regex'] = [re.compile(regex) for regex in configs['sys_msg']['regex']]
        configs['sys_msg']['anonymisation'] = [re.compile(regex) for regex in configs['sys_msg']['anonymisation']]
    if 'spacy' in configs:
        configs['spacy']['model'] = spacy.load(configs['spacy']['model'])
    if 'stanza' in configs:
        stanza.download(configs['stanza']['model'])
        configs['stanza']['model'] = stanza.Pipeline(configs['stanza']['model'])
    for entity_type in configs['anonymisation']:
        if 'regex' in configs['anonymisation'][entity_type]:
            configs['anonymisation'][entity_type]['regex'] = re.compile(configs['anonymisation'][entity_type]['regex'])
    # Get input paths
    input_file_path: str = args.input_file_path
    input_file: str
    input_file_name: str
    input_file_dir_path, input_file_name = os.path.split(input_file_path)
    input_file_name, _ = os.path.splitext(input_file_name)
    # Load data and get target columns
    with open(input_file_path) as f:
        data: List[Dict] = json.load(f)
    # Preprocessing
    # Process data
    # Do anonymisation (if required)
    if configs['output_anonymisation']:
        #
        logging.info("Starting anonymisation")
        # Run anonymisation and save result in data frame
        # Standardise corpus
        with parallel_backend(configs.get('parallel_backend', 'threading'), n_jobs=configs.get('n_jobs', -1)):
            data = Parallel(verbose=configs.get('verbosity_level', 1))(
                delayed(anonymise_sample)(sample, configs) for sample in data
            )
        df = pd.DataFrame(data)
        # Serialise output
        # Get output path
        output_file_name: str = f'{input_file_name}_anonymised_{date_time}.csv'
        output_file_path: str = os.path.join(input_file_dir_path, output_file_name)
        # Serialise data
        df.to_csv(output_file_path, index=False)
        #
        logging.info(f"Anonymisation concluded, output saved at {output_file_path}")
    # Get entity data and stats (if required)
    if configs['output_entities_data'] or configs['output_entities_data']:
        #
        logging.info("Starting entities data extraction and statistics computation")
        # Extract stats and info
        entities_data: Dict[str, List[Dict[str, Dict[str, List[str]]]]] = {
            key: list() for key in configs['tgt_columns']
        }
        entities_stats: Dict[str, List[Dict[str, Dict[str, int]]]] = {key: list() for key in configs['tgt_columns']}
        with parallel_backend(configs.get('parallel_backend', 'threading'), n_jobs=configs.get('n_jobs', -1)):
            tmp_data = Parallel(verbose=configs.get('verbosity_level', 1))(
                delayed(extract_entity_info_sample)(sample, configs) for sample in data
            )
        for data_dict in tqdm(tmp_data):
            for key in configs['tgt_columns']:
                tmp_entities_data, tmp_entities_stats = data_dict[key]
                entities_data[key].append(tmp_entities_data)
                entities_stats[key].append(tmp_entities_stats)
        # Serialise output (if required)
        if configs['output_entities_data']:
            # Get entities data file path
            entities_data_file_name: str = f'{input_file_name}_entities_{date_time}.json'
            entities_data_file_path: str = os.path.join(input_file_dir_path, entities_data_file_name)
            # Serialise data
            with open(entities_data_file_path, 'w') as f:
                json.dump(entities_data, f)
        if configs['output_entities_data']:
            # Get entities stats file path
            entities_data_file_name: str = f'{input_file_name}_stats_{date_time}.json'
            entities_data_file_path: str = os.path.join(input_file_dir_path, entities_data_file_name)
            # Serialise stats
            with open(entities_data_file_path, 'w') as f:
                json.dump(entities_stats, f)
        #
        logging.info(f"Entities data and statistics computed")
    # Conclude script
    logging.info("Process completed successfully")

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser()
    # Add arguments to parser
    args_parser.add_argument(
        '--config_file_path',
        type=str,
        help="Path to the YAML file containing the configuration for the execution."
    )
    args_parser.add_argument(
        '--input_file_path',
        type=str,
        help="Path to the JSON file with the data to anonymise."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
