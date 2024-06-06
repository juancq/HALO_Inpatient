import os
import torch
import numpy as np
import polars as pl
import random
import pickle
from pathlib import Path
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from halo_data_module import HALODataModule
from lightning.pytorch.callbacks import RichProgressBar

from model import HALOModel
from config import HALOConfig

def main():
    SEED = 4
    seed_everything(SEED)
    torch.set_float32_matmul_precision('high')

    dv = pl.read_excel("df_vars_new.xlsx")
    config = HALOConfig(dv)

    # Initialize data module
    # set data path
    data_path = Path('/mnt/data_volume/apdc/lumos')
    # must implement your own data module, example provided in halo_data_module
    data_module = HALODataModule(config, batch_size=config.batch_size, 
                    train_name=data_path / 'df_trn_ppn.parquet',
                    val_name=data_path / 'df_val_ppn.parquet',
                    train_split=0.5,
                    )

    # Initialize model
    model = HALOModel(config)
    model.transformer = torch.compile(model.transformer)
    model.ehr_head = torch.compile(model.ehr_head)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./save/',
        filename='halo_model-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        mode='min',
    )

    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,  # number of epochs with no improvement after which training will be stopped
        mode='min'
    )

    # Trainer
    trainer = Trainer(
        max_epochs=config.epoch,
        callbacks=[checkpoint_callback, early_stopping_callback,
                RichProgressBar(leave=True)],
        # add mlflow here
        #logger=True,
    )

    # Train the model
    trainer.fit(model, data_module)

    # Evaluate the model
    # need to add test set to data module in order to call this
    #trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()
