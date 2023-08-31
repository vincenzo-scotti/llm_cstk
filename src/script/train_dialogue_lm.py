import sys
import tempfile
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


CALLBACKS: Dict = {
    'early_stopping': pl.callbacks.EarlyStopping, 'checkpoint_callback': pl.callbacks.ModelCheckpoint
}


# TODO integrate better this step
def quantised_trainin(model: DialogueLM, data_splits: Dict[str, ChatDataset], configs: Dict):
    ...


def main(args: Namespace):
    # Init environment
    configs: Dict = init_training_environment(args.config_file_path)
    # Start Logging info
    logging.info("Script started and configuration file loaded")
    # Load NN Estimator
    model: DialogueLM = DialogueLM(**configs['dialogue_lm'])
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
        **configs['trainer'],
        callbacks=list(callbacks.values()),
        logger=loggers
    )
    logging.info("Trainer instantiated")
    # Train neural network
    start_time = datetime.now()
    logging.info("Training started")
    trainer.fit(model, train_dataloaders=data_loaders['train'], val_dataloaders=data_loaders['validation'])
    stop_time = datetime.now()
    logging.info(f"Training completed (elapsed time: {stop_time - start_time})")
    # Restore best weights (if required)
    if 'checkpoint_callback' in configs.get('callbacks', dict()):
        # Load torch checkpoint
        checkpoint = torch.load(callbacks['checkpoint_callback'].best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info(f"Best checkpoint restored from {callbacks['checkpoint_callback'].best_model_path}")
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
