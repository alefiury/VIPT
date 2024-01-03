from __future__ import annotations

import os
import warnings
from logging import getLogger
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.accelerators import MPSAccelerator, TPUAccelerator
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.tuner import Tuner
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from modules.model import VIPT
from modules.descriminators import MultiPeriodDiscriminator
from modules.commons import slice_segments, clip_grad_value_
from modules.mel_processing import mel_spectrogram_torch
from modules.losses import (kl_loss, feature_loss, discriminator_loss, generator_loss, prosody_kl_loss)
import modules.utils as utils
from modules.mel_processing import spec_to_mel_torch

LOG = getLogger(__name__)
torch.set_float32_matmul_precision("high")


class VitsLightning(pl.LightningModule):
    def __init__(self, reset_optimizer: bool = False, **hparams: Any):
        super().__init__()
        self._temp_epoch = 0  # Add this line to initialize the _temp_epoch attribute
        self.save_hyperparameters("reset_optimizer")
        self.save_hyperparameters(*[k for k in hparams.keys()])
        torch.manual_seed(self.hparams.train.seed)
        self.automatic_optimization = False

        self.net_g = VIPT(
            self.hparams.data.filter_length // 2 + 1,
            self.hparams.train.segment_size // self.hparams.data.hop_length,
            **self.hparams.model,
        )

        self.net_d = MultiPeriodDiscriminator(self.hparams.model.use_spectral_norm)

        self.learning_rate = self.hparams.train.learning_rate

        self.optim_g = torch.optim.AdamW(
            self.net_g.parameters(),
            self.learning_rate,
            betas=self.hparams.train.betas,
            eps=self.hparams.train.eps,
        )

        self.optim_d = torch.optim.AdamW(
            self.net_d.parameters(),
            self.learning_rate,
            betas=self.hparams.train.betas,
            eps=self.hparams.train.eps,
        )

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=self.hparams.train.lr_decay
        )

        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_d, gamma=self.hparams.train.lr_decay
        )

        self.optimizers_count = 2
        self.tuning = False

    def on_train_start(self) -> None:
        if not self.tuning:
            self.set_current_epoch(self._temp_epoch)
            total_batch_idx = self._temp_epoch * len(self.trainer.train_dataloader)
            self.set_total_batch_idx(total_batch_idx)
            global_step = total_batch_idx * self.optimizers_count
            self.set_global_step(global_step)

        # check if using tpu or mps
        if isinstance(self.trainer.accelerator, (TPUAccelerator, MPSAccelerator)):
            # patch torch.stft to use cpu
            LOG.warning("Using TPU/MPS. Patching torch.stft to use cpu.")

            def stft(
                input: torch.Tensor,
                n_fft: int,
                hop_length: int | None = None,
                win_length: int | None = None,
                window: torch.Tensor | None = None,
                center: bool = True,
                pad_mode: str = "reflect",
                normalized: bool = False,
                onesided: bool | None = None,
                return_complex: bool | None = None,
            ) -> torch.Tensor:
                device = input.device
                input = input.cpu()
                if window is not None:
                    window = window.cpu()
                return torch.functional.stft(
                    input,
                    n_fft,
                    hop_length,
                    win_length,
                    window,
                    center,
                    pad_mode,
                    normalized,
                    onesided,
                    return_complex,
                ).to(device)

            torch.stft = stft

        elif "bf" in self.trainer.precision:
            LOG.warning("Using bf. Patching torch.stft to use fp32.")

            def stft(
                input: torch.Tensor,
                n_fft: int,
                hop_length: int | None = None,
                win_length: int | None = None,
                window: torch.Tensor | None = None,
                center: bool = True,
                pad_mode: str = "reflect",
                normalized: bool = False,
                onesided: bool | None = None,
                return_complex: bool | None = None,
            ) -> torch.Tensor:
                dtype = input.dtype
                input = input.float()
                if window is not None:
                    window = window.float()
                return torch.functional.stft(
                    input,
                    n_fft,
                    hop_length,
                    win_length,
                    window,
                    center,
                    pad_mode,
                    normalized,
                    onesided,
                    return_complex,
                ).to(dtype)

            torch.stft = stft

    def on_train_end(self) -> None:
        self.save_checkpoints(adjust=0)

    def save_checkpoints(self, adjust=1):
        if self.tuning or self.trainer.sanity_checking:
            return

        # only save checkpoints if we are on the main device
        if (
            hasattr(self.device, "index")
            and self.device.index != None
            and self.device.index != 0
        ):
            return

        # `on_train_end` will be the actual epoch, not a -1, so we have to call it with `adjust = 0`
        current_epoch = self.current_epoch + adjust
        total_batch_idx = self.total_batch_idx - 1 + adjust

        utils.save_checkpoint(
            self.net_g,
            self.optim_g,
            self.learning_rate,
            current_epoch,
            Path(self.hparams.model_dir)
            / f"G_{total_batch_idx if self.hparams.train.get('ckpt_name_by_step', False) else current_epoch}.pth",
        )

        utils.save_checkpoint(
            self.net_d,
            self.optim_d,
            self.learning_rate,
            current_epoch,
            Path(self.hparams.model_dir)
            / f"D_{total_batch_idx if self.hparams.train.get('ckpt_name_by_step', False) else current_epoch}.pth",
        )

        keep_ckpts = self.hparams.train.get("keep_ckpts", 0)

        if keep_ckpts > 0:
            utils.clean_checkpoints(
                path_to_models=self.hparams.model_dir,
                n_ckpts_to_keep=keep_ckpts,
                sort_by_time=True,
            )

    def set_current_epoch(self, epoch: int):
        LOG.info(f"Setting current epoch to {epoch}")
        self.trainer.fit_loop.epoch_progress.current.completed = epoch
        self.trainer.fit_loop.epoch_progress.current.processed = epoch
        assert self.current_epoch == epoch, f"{self.current_epoch} != {epoch}"

    def set_global_step(self, global_step: int):
        LOG.info(f"Setting global step to {global_step}")
        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed = (
            global_step
        )
        self.trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.total.completed = (
            global_step
        )
        assert self.global_step == global_step, f"{self.global_step} != {global_step}"

    def set_total_batch_idx(self, total_batch_idx: int):
        LOG.info(f"Setting total batch idx to {total_batch_idx}")
        self.trainer.fit_loop.epoch_loop.batch_progress.total.ready = (
            total_batch_idx + 1
        )
        self.trainer.fit_loop.epoch_loop.batch_progress.total.completed = (
            total_batch_idx
        )
        assert (
            self.total_batch_idx == total_batch_idx + 1
        ), f"{self.total_batch_idx} != {total_batch_idx + 1}"

    @property
    def total_batch_idx(self) -> int:
        return self.trainer.fit_loop.epoch_loop.total_batch_idx + 1

    def load(self, reset_optimizer: bool = False):
        latest_g_path = utils.latest_checkpoint_path(self.hparams.model_dir, "G_*.pth")
        latest_d_path = utils.latest_checkpoint_path(self.hparams.model_dir, "D_*.pth")
        if latest_g_path is not None and latest_d_path is not None:
            try:
                _, _, _, epoch = utils.load_checkpoint(
                    latest_g_path,
                    self.net_g,
                    self.optim_g,
                    reset_optimizer,
                )
                _, _, _, epoch = utils.load_checkpoint(
                    latest_d_path,
                    self.net_d,
                    self.optim_d,
                    reset_optimizer,
                )
                self._temp_epoch = epoch
                self.scheduler_g.last_epoch = epoch - 1
                self.scheduler_d.last_epoch = epoch - 1
            except Exception as e:
                raise RuntimeError("Failed to load checkpoint") from e
        else:
            LOG.warning("No checkpoint found. Start from scratch.")

    def configure_optimizers(self):
        return [self.optim_g, self.optim_d], [self.scheduler_g, self.scheduler_d]

    def log_image_dict(
        self, image_dict: dict[str, Any], dataformats: str = "HWC"
    ) -> None:
        if not isinstance(self.logger, TensorBoardLogger):
            warnings.warn("Image logging is only supported with TensorBoardLogger.")
            return
        writer: SummaryWriter = self.logger.experiment
        for k, v in image_dict.items():
            try:
                writer.add_image(k, v, self.total_batch_idx, dataformats=dataformats)
            except Exception as e:
                warnings.warn(f"Failed to log image {k}: {e}")

    def log_audio_dict(self, audio_dict: dict[str, Any]) -> None:
        if not isinstance(self.logger, TensorBoardLogger):
            warnings.warn("Audio logging is only supported with TensorBoardLogger.")
            return
        writer: SummaryWriter = self.logger.experiment
        for k, v in audio_dict.items():
            writer.add_audio(
                k,
                v.float(),
                self.total_batch_idx,
                sample_rate=self.hparams.data.sampling_rate,
            )

    def log_dict_(self, log_dict: dict[str, Any], **kwargs) -> None:
        if not isinstance(self.logger, TensorBoardLogger):
            warnings.warn("Logging is only supported with TensorBoardLogger.")
            return
        writer: SummaryWriter = self.logger.experiment
        for k, v in log_dict.items():
            writer.add_scalar(k, v, self.total_batch_idx)
        kwargs["logger"] = False
        self.log_dict(log_dict, **kwargs)

    def log_(self, key: str, value: Any, **kwargs) -> None:
        self.log_dict_({key: value}, **kwargs)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self.net_g.train()
        self.net_d.train()

        # get optims
        optim_g, optim_d = self.optimizers()

        # Generator
        # train
        self.toggle_optimizer(optim_g)
        x, x_lengths, sid, lid, y, y_lengths = batch
        output_dict = self.net_g(x, x_lengths, sid, lid, y, y_lengths)
        mel = spec_to_mel_torch(
            y,
            self.hparams
        )
        y_mel = slice_segments(
            mel,
            output_dict["ids_slice"],
            self.hparams.train.segment_size // self.hparams.data.hop_length,
        )
        y_hat_mel = mel_spectrogram_torch(output_dict["model_outputs"].float().squeeze(1), self.hparams)
        y_mel = y_mel[..., : y_hat_mel.shape[-1]]
        y = slice_segments(
            y,
            output_dict["ids_slice"] * self.hparams.data.hop_length,
            self.hparams.train.segment_size,
        )
        y = y[..., : output_dict["model_outputs"].shape[-1]]

        # generator loss
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, output_dict["model_outputs"])

        with autocast(enabled=False):
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.train.c_mel
            loss_kl = (
                kl_loss(output_dict["z_p"], output_dict["logs_q"], output_dict["m_p"], output_dict["logs_p"], output_dict["z_mask"]) * self.hparams.train.c_kl
            )
            loss_prosody_kl = prosody_kl_loss(output_dict["pros_mu"], output_dict["pros_logvar"], self.hparams.train.batch_size) * self.hparams.train.c_prosody_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_prosody_kl

        # log loss
        self.log_("lr", self.optim_g.param_groups[0]["lr"])
        self.log_dict_(
            {
                "loss/g/total": loss_gen_all,
                "loss/g/fm": loss_fm,
                "loss/g/mel": loss_mel,
                "loss/g/kl": loss_kl,
                "loss/g/prosody_kl": loss_prosody_kl,
            },
            prog_bar=True,
        )

        if self.total_batch_idx % self.hparams.train.log_interval == 0:
            self.log_image_dict(
                {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().float().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().float().numpy()
                    ),
                }
            )

        accumulate_grad_batches = self.hparams.train.get("accumulate_grad_batches", 1)
        should_update = (
            batch_idx + 1
        ) % accumulate_grad_batches == 0 or self.trainer.is_last_batch
        # optimizer
        self.manual_backward(loss_gen_all / accumulate_grad_batches)
        if should_update:
            self.log_(
                "grad_norm_g", clip_grad_value_(self.net_g.parameters(), None)
            )
            optim_g.step()
            optim_g.zero_grad()
        self.untoggle_optimizer(optim_g)

        # Discriminator
        # train
        self.toggle_optimizer(optim_d)
        y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, output_dict["model_outputs"].detach())

        # discriminator loss
        with autocast(enabled=False):
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc

        # log loss
        self.log_("loss/d/total", loss_disc_all, prog_bar=True)

        # optimizer
        self.manual_backward(loss_disc_all / accumulate_grad_batches)
        if should_update:
            self.log_(
                "grad_norm_d", clip_grad_value_(self.net_d.parameters(), None)
            )
            optim_d.step()
            optim_d.zero_grad()
        self.untoggle_optimizer(optim_d)

        # end of epoch
        if self.trainer.is_last_batch:
            self.scheduler_g.step()
            self.scheduler_d.step()

    def validation_step(self, batch, batch_idx):
        # avoid logging with wrong global step
        if self.global_step == 0:
            return
        with torch.no_grad():
            self.net_g.eval()
            x, x_lengths, sid, lid, y, y_lengths = batch
            y_hat = self.net_g.infer(x, x_lengths, sid, lid)
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1).float(), self.hparams)
            self.log_audio_dict(
                {f"gen/audio_{batch_idx}": y_hat[0], f"gt/audio_{batch_idx}": y[0]}
            )
            self.log_image_dict(
                {
                    "gen/mel": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].cpu().float().numpy()
                    ),
                    "gt/mel": utils.plot_spectrogram_to_numpy(
                        y[0].cpu().float().numpy()
                    ),
                }
            )

    def on_validation_end(self) -> None:
        self.save_checkpoints()