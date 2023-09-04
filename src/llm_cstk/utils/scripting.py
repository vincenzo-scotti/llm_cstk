import os
import sys
import logging
from shutil import copy2
from datetime import datetime

import torch
import pytorch_lightning as pl

import yaml

try:
    from transformers import BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from trl import SFTTrainer
except ImportError:
    pass

from typing import Dict


def _preprocess_quantised_training_configs(configs: Dict) -> Dict:
    # B&B
    bnb_configs: Dict = configs['quantised_trainer'].pop('bnb')
    bnb_4bit_compute_dtype = bnb_configs.pop('bnb_4bit_compute_dtype', 'bfloat16')
    if bnb_4bit_compute_dtype == 'bfloat16':
        bnb_4bit_compute_dtype = torch.bfloat16
    elif bnb_4bit_compute_dtype == 'float16':
        bnb_4bit_compute_dtype = torch.float16
    elif bnb_4bit_compute_dtype == 'float32':
        bnb_4bit_compute_dtype = torch.float32
    else:
        raise ValueError(f"Unknown value for `bnb_4bit_compute_dtype`: `{bnb_4bit_compute_dtype}`")
    bnb_configs['bnb_4bit_compute_dtype'] = bnb_4bit_compute_dtype
    configs['quantised_trainer']['bnb'] = BitsAndBytesConfig(**bnb_configs)
    # Lora
    configs['quantised_trainer']['lora'] = LoraConfig(**configs['quantised_trainer']['lora'])
    # Training arguments
    configs['quantised_trainer']['training_args'] = TrainingArguments(
        output_dir=configs['current_experiment_q_dir_path'],
        logging_dir=configs['current_experiment_q_dir_path'],
        **configs['quantised_trainer']['training_args']
    )
    # Merging (after training)
    merging_torch_dtype = configs['quantised_trainer']['merge'].pop('torch_dtype', 'bfloat16')
    if merging_torch_dtype == 'bfloat16':
        merging_torch_dtype = torch.bfloat16
    elif merging_torch_dtype == 'float16':
        merging_torch_dtype = torch.float16
    elif merging_torch_dtype == 'float32':
        merging_torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown value for `torch_dtype`: `{merging_torch_dtype}`")
    configs['quantised_trainer']['merge']['torch_dtype'] = merging_torch_dtype
    # Dialogue LM
    configs['dialogue_lm'].update(
        model__quantization_config=configs['quantised_trainer']['bnb'],
        model__use_cache=False,
        model__device_map='auto'
    )
    """
    configs['huggingface'] = dict()
    configs['huggingface']['transformer'] = configs['dialogue_lm']['transformer']
    configs['huggingface']['model'] = {
        key.split('__', 1)[1]: configs['dialogue_lm'][key]
        for key in configs['dialogue_lm']
        if key.startswith('model')
    } | {
        'quantization_config': configs['quantised_trainer']['bnb'],
        'use_cache': False,
        'device_map': 'auto'
    }
    configs['huggingface']['tokeniser'] = {
        key.split('__', 1)[1]: configs['dialogue_lm'][key]
        for key in configs['dialogue_lm']
        if key.startswith('tokeniser')
    }
    """

    return configs


def init_training_environment(config_file_path: str) -> Dict:
    # Get datetime string
    date_time: str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # Load configs
    with open(config_file_path) as f:
        configs: Dict = yaml.full_load(f)
    # Paths and directories
    experiments_dir_path: str = os.path.join(configs['experiments_dir_path'])
    if not os.path.exists(experiments_dir_path):
        os.mkdir(experiments_dir_path)
    experiment_series_dir_path: str = os.path.join(experiments_dir_path, configs['experiment_series'])
    if not os.path.exists(experiment_series_dir_path):
        os.mkdir(experiment_series_dir_path)
    current_experiment_dir_path: str = os.path.join(
        experiment_series_dir_path, f"{configs['experiment_id']}_{date_time}"
    )
    configs['current_experiment_dir_path'] = current_experiment_dir_path
    if not os.path.exists(current_experiment_dir_path):
        os.mkdir(current_experiment_dir_path)
    # Set random seed
    pl.seed_everything(configs.get('random_seed'), workers=True)
    # Dump configs
    config_dump_file_path = os.path.join(current_experiment_dir_path, f'config.yml')
    copy2(config_file_path, config_dump_file_path)
    # Init logging
    if configs.get('log_file', False):
        log_file_path = os.path.join(current_experiment_dir_path, f'training.log')
    else:
        log_file_path = None
    logging.basicConfig(filename=log_file_path, level=configs.get('log_level', 'INFO'))
    # Additional quantisation configs
    if 'quantised_trainer' in configs:
        current_experiment_q_dir_path: str = os.path.join(
            experiment_series_dir_path, f"{configs['experiment_id']}_QLoRA_{date_time}"
        )
        configs['current_experiment_q_dir_path'] = current_experiment_q_dir_path
        if not os.path.exists(current_experiment_q_dir_path):
            os.mkdir(current_experiment_q_dir_path)
        configs = _preprocess_quantised_training_configs(configs)

    return configs


def init_rest_server_environment(config_file_path: str) -> Dict:
    # Get datetime string
    date_time: str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    # Load configs
    with open(config_file_path) as f:
        configs: Dict = yaml.full_load(f)
    # Paths and directories
    services_dir_path: str = os.path.join(configs['services_dir_path'])
    if not os.path.exists(services_dir_path):
        os.mkdir(services_dir_path)
    service_dir_path: str = os.path.join(services_dir_path, configs['service'])
    if not os.path.exists(service_dir_path):
        os.mkdir(service_dir_path)
    current_service_instance_dir_path: str = os.path.join(
        service_dir_path, f"{date_time}"
    )
    configs['current_service_instance_dir_path'] = current_service_instance_dir_path
    if not os.path.exists(current_service_instance_dir_path):
        os.mkdir(current_service_instance_dir_path)
    # Set random seed
    pl.seed_everything(configs.get('random_seed'), workers=True)
    # Dump configs
    config_dump_file_path = os.path.join(current_service_instance_dir_path, f'config.yml')
    copy2(config_file_path, config_dump_file_path)
    # Init logging
    if configs.get('log_file', False):
        log_file_path = os.path.join(current_service_instance_dir_path, f"{configs['service']}_service.log")
    else:
        log_file_path = None
    logging.basicConfig(filename=log_file_path, level=configs.get('log_level', 'INFO'))

    return configs
