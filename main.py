import os

import torch.nn as nn
import numpy as np
# Load yaml
from omegaconf import OmegaConf
import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from modules.model import VIPT
from data.ljspeech import ljspeech
from lightning.pytorch.callbacks import DeviceStatsMonitor, RichProgressBar
from data.text.cleaners import english_cleaners2
from data.text.utils import symbols_to_ids, ids_to_symbols
from data.text.symbols import symbols_to_ids_dict
from data.dataloader import TextAudioLoader, TextAudioCollate
from train import VIPTLightning


def main():
    # Load config
    config = OmegaConf.load("configs/default.yaml")


    # print(config)

    # print([config[b] for b in config.keys()])

    # print(config)
    # # Initialize the model
    # model = VIPT(
    #     **config.model,
    #     n_vocab=config.n_vocab,
    # )

    # # X vales must be in the range of 0 ~ n_vocab and the tipe LongTensor
    # x = torch.randint(0, config.n_vocab, (2, 100)).long()
    # x_lengths = torch.tensor([100, 100])
    # sid = torch.tensor([0, 1]).long()
    # lid = torch.tensor([0, 1]).long()

    # # y with the same dim of a spectrogram
    # y = torch.randn(2, 80, 100)
    # y_lengths = torch.tensor([100, 100])

    # # sid = torch.tensor([[b] for b in sid])
    # # lid = torch.tensor([[b] for b in lid])

    # print(sid.shape)
    # print(lid.shape)

    # # emb_g = nn.Embedding(2, 512)

    # # sid_emb = emb_g(sid)

    # # print(sid_emb.shape)

    # # exit()

    # # Forward pass
    # y_hat = model(x, x_lengths, sid, lid, y, y_lengths)

    # print(y_hat["model_outputs"].shape)

    metadata = ljspeech("/media/alefiury/2 TB/Projetos/Alcateia/datasets/LJSpeech-1.1")

    train_rate = 0.9
    val_rate = 0.1

    train_metadata = metadata[:int(len(metadata) * train_rate)]
    val_metadata = metadata[int(len(metadata) * train_rate):]

    train_dataset = TextAudioLoader(train_metadata, config.data)
    val_dataset = TextAudioLoader(val_metadata, config.data)
    collate_fn = TextAudioCollate()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.train.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.train.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model_pl = VIPTLightning(hparams=config)

    print(model_pl)

    logger = WandbLogger(project="VIPT", name="VIPT-test", entity="alefiury")

    trainer = pl.Trainer(
        logger=logger,
        val_check_interval=config.train.eval_interval,
        max_epochs=config.train.epochs,
        check_val_every_n_epoch=None,
        precision="16-mixed" if config.train.fp16_run else "bf16-mixed" if config.train.get("bf16_run", False) else 32,
        callbacks=[RichProgressBar(), DeviceStatsMonitor()],
        benchmark=True,
        enable_checkpointing=False,
        devices=[0],
    )

    trainer.fit(model_pl, train_loader, val_loader)



if __name__ == '__main__':
    main()