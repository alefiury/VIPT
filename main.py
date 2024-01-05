import argparse

import torch
import lightning as pl
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import DeviceStatsMonitor, RichProgressBar

from train import VIPTLightning
from data.ljspeech import ljspeech
from data.dataloader import TextAudioLoader, TextAudioCollate


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("-g", "--gpu", type=int, default=6)
    args = parser.parse_args()
    # Load config
    config = OmegaConf.load(args.config)

    metadata = ljspeech("/hadatasets/alef.ferreira/translation/data/LJSpeech-1.1")

    val_rate = 0.01
    train_rate = 1 - val_rate

    train_metadata = metadata[:int(len(metadata) * train_rate)]
    val_metadata = metadata[int(len(metadata) * train_rate):]

    print(f"Train metadata: {len(train_metadata)}")
    print(f"Val metadata: {len(val_metadata)}")

    train_dataset = TextAudioLoader(train_metadata, config.data)
    val_dataset = TextAudioLoader(val_metadata, config.data)
    collate_fn = TextAudioCollate()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model_pl = VIPTLightning(hparams=config)

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
        devices=[args.gpu],
        num_sanity_val_steps=2
    )

    trainer.fit(model_pl, train_loader, val_loader)


if __name__ == '__main__':
    main()