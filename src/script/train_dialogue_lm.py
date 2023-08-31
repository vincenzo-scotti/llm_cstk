import sys
from argparse import ArgumentParser, Namespace
import logging
from datetime import datetime

from typing import Dict

import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from llm_cstk.model import DialogueLM
from llm_cstk.data import ChatDataset
from llm_cstk.utils.scripting import init_training_environment

quantised_training_available: bool = True
try:
    from transformers import BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from peft import AutoPeftModelForCausalLM, AutoPeftModelForSeq2SeqLM
    from trl import SFTTrainer
except ImportError:
    quantised_training_available = False


CALLBACKS: Dict = {
    'early_stopping': pl.callbacks.EarlyStopping, 'checkpoint_callback': pl.callbacks.ModelCheckpoint
}

# NOTE: all steps can be integrated in the PL interface, dropping the need for separate trainers and so on


def main(args: Namespace):
    # Init environment
    configs: Dict = init_training_environment(args.config_file_path)
    # Start logging info
    logging.info("Script started and configuration file loaded")
    # Check training type
    if 'trainer' in configs:
        quantised_training: bool = False
    elif 'quantised_trainer' in configs:
        if quantised_training_available:
            quantised_training = True
        else:
            raise RuntimeError("Quantised training is not available on this machine.")
    else:
        raise ValueError(
            "Missing training configs. "
            "You must specify one between `trainer` and `quantised_trainer` in the configuration file."
        )
    # Load NN Estimator
    model: DialogueLM = DialogueLM(**configs['dialogue_lm'])
    if quantised_training:
        # Complete model preparation
        model._language_model.config.pretraining_tp = 1
        model._language_model = prepare_model_for_kbit_training(model)
        model._language_model = get_peft_model(model, configs['lora'])
    # Start Logging info
    logging.info("Neural network loaded")
    # Create data set splits
    data_splits: Dict[str, ChatDataset] = {
        split: ChatDataset(
            split,
            **configs['data']['params']
        )
        for split in configs['data']['splits']
    }
    logging.info("Data set splits loaded")
    # Create data loaders
    data_loaders: Dict[str, DataLoader] = {
        split: DataLoader(data, collate_fn=data.collate, shuffle=split == 'train', **configs['data']['loader'][split])
        for split, data in data_splits.items()
    }
    model.set_n_training_steps(len(data_loaders['train']) / configs['trainer'].get('accumulate_grad_batches', 1))
    logging.info("Data loaders instantiated")
    # Create callbacks
    callbacks = {
        callback_id: CALLBACKS[callback_id](**callback_configs)
        for callback_id, callback_configs in configs.get('callbacks', dict()).items()
    } | {'learning_rate_callback': pl.callbacks.LearningRateMonitor()}
    q_callbacks = {}
    logging.info("Callbacks instantiated")
    # Create loggers
    loggers = [
        pl.loggers.TensorBoardLogger(configs['current_experiments_dir_path']),
        pl.loggers.CSVLogger(configs['current_experiments_dir_path'])
    ]
    logging.info("Loggers instantiated")
    # Instantiate Trainer object with the callbacks
    trainer = pl.Trainer(
        default_root_dir=configs['current_experiments_dir_path'],
        **configs.get('trainer', dict()),
        callbacks=list(callbacks.values()),
        logger=loggers
    )
    q_trainer = SFTTrainer(
        model=model._language_model,
        train_dataset=data_splits['train'],
        eval_dataset=data_splits['validation'],
        peft_config=configs['quantised_trainer']['lora'],
        max_seq_length=model._tokeniser.model_max_length,
        tokenizer=model._tokeniser,
        packing=True,
        args=configs['quantised_trainer']['training_args'],
        data_collator=data_splits['train'].huggingface_collate
    ) if quantised_training else None
    logging.info("Trainer instantiated")
    # Train neural network
    start_time = datetime.now()
    logging.info("Training started")
    if quantised_training:
        q_trainer.train()
        q_trainer.save_model()
    else:
        trainer.fit(model, train_dataloaders=data_loaders['train'], val_dataloaders=data_loaders['validation'])
    stop_time = datetime.now()
    logging.info(f"Training completed (elapsed time: {stop_time - start_time})")
    # Restore best weights (if required)
    if 'checkpoint_callback' in configs.get('callbacks', dict()) and not quantised_training:
        # Load torch checkpoint
        checkpoint = torch.load(callbacks['checkpoint_callback'].best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info(f"Best checkpoint restored from {callbacks['checkpoint_callback'].best_model_path}")
    elif quantised_training:
        # Load and merge weights
        if model._language_model.config.is_encoder_decoder:
            model._language_model = AutoPeftModelForSeq2SeqLM.from_pretrained(
                configs['current_experiment_q_dir_path'], **configs['quantised_trainer']['merge']
            ).merge_and_unload()
        else:
            model._language_model = AutoPeftModelForCausalLM.from_pretrained(
                configs['current_experiment_q_dir_path'], **configs['quantised_trainer']['merge']
            ).merge_and_unload()
    # Activate metrics for final evaluation
    model.enable_metrics()
    # Change padding side for generator evaluation (allows multiple sequences)
    model.invert_padding_side()
    # Test neural network
    start_time = datetime.now()
    logging.info("Validation started")
    trainer.validate(model, dataloaders=data_loaders['validation'])
    stop_time = datetime.now()
    logging.info(f"Validation completed (elapsed time: {stop_time - start_time})")
    start_time = datetime.now()
    logging.info("Testing started")
    trainer.test(model, dataloaders=data_loaders['test'])
    stop_time = datetime.now()
    logging.info(f"Testing completed (elapsed time: {stop_time - start_time})")
    # Change padding side for generator evaluation (allows multiple sequences)
    model.invert_padding_side()
    # Save model (if required)
    if 'model_dir_path' in configs:
        if quantised_training:
            model._language_model.save_pretrained(configs['model_dir_path'], safe_serialization=True)
            model._tokeniser.save_pretrained(configs['model_dir_path'])
        else:
            model.save(configs['model_dir_path'])
        logging.info(f"Model saved at \'{configs['model_dir_path']}\'")
    # Apply post-training quantisation
    if 'quantisation' in configs:
        start_time = datetime.now()
        logging.info("Quantisation started")
        model.quantise(
            configs['quantisation'].get('bits', 8),
            data_splits[configs['quantisation'].get('split', 'train')].as_strings(),
            path=configs.get('model_dir_path'),
            **configs['quantisation'].get('params', dict())
        )
        stop_time = datetime.now()
        logging.info(f"Quantisation completed (elapsed time: {stop_time - start_time})")
        # Save quantised model
        model.save(configs['quantisation']['model_dir_path'])
        logging.info(f"Quantised model saved at \'{configs['quantisation']['model_dir_path']}\'")
    # Close script info
    logging.info("Script executed successfully")

    return 0


if __name__ == "__main__":
    # Instantiate argument parser
    args_parser: ArgumentParser = ArgumentParser(
        prog='dialogue_lm_trainer',
        description='Script to train transformer-based dialogue language models'
    )
    # Add arguments to parser
    args_parser.add_argument(
        '--config_file_path',
        type=str,
        help="Path to the YAML file containing the configuration for the experiment."
    )
    # Run experiment
    main(args_parser.parse_args(sys.argv[1:]))
