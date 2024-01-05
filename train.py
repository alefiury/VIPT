from typing import Dict
from pathlib import Path
from logging import getLogger

import torch
import wandb
import lightning.pytorch as pl
from torch.cuda.amp import autocast
from torch.nn import functional as F
from lightning.pytorch.accelerators import MPSAccelerator, TPUAccelerator

import modules.utils as utils
from modules.model import VIPT
from data.text.symbols import symbols
from modules.descriminators import MultiPeriodDiscriminator
from modules.mel_processing import spec_to_mel_torch
from modules.mel_processing import mel_spectrogram_torch
from modules.commons import slice_segments, clip_grad_value_
from modules.losses import (kl_loss, feature_loss, discriminator_loss, generator_loss, prosody_kl_loss)

LOG = getLogger(__name__)
torch.set_float32_matmul_precision("high")


class VIPTLightning(pl.LightningModule):
    def __init__(self, reset_optimizer: bool = False, hparams: Dict = None):
        super().__init__()
        self._temp_epoch = 0  # Add this line to initialize the _temp_epoch attribute
        self.save_hyperparameters(reset_optimizer)
        self.save_hyperparameters(hparams)

        torch.manual_seed(self.hparams.train.seed)
        self.automatic_optimization = False
        self.net_g = VIPT(
            **self.hparams.model,
            segment_size=self.hparams.train.segment_size // self.hparams.data.hop_length,
            n_vocab=len(symbols)
        )

        self.net_d = MultiPeriodDiscriminator(self.hparams.data.use_spectral_norm)

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

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self.net_g.train()
        self.net_d.train()

        # get optims
        optim_g, optim_d = self.optimizers()

        # Generator
        # train
        self.toggle_optimizer(optim_g)
        x, x_lengths, y, y_lengths, wav, wav_lengths, sid, lid = batch
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
        wav = slice_segments(
            wav,
            output_dict["ids_slice"] * self.hparams.data.hop_length,
            self.hparams.train.segment_size,
        )
        wav = wav[..., : output_dict["model_outputs"].shape[-1]]

        # generator loss
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(wav, output_dict["model_outputs"])

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
        self.log("lr", self.optim_g.param_groups[0]["lr"])
        self.log_dict(
            {
                "loss-g_total": loss_gen_all,
                "loss-g_fm": loss_fm,
                "loss-g_mel": loss_mel,
                "loss-g_kl": loss_kl,
                "loss-g_prosody_kl": loss_prosody_kl,
            },
            prog_bar=True,
        )

        if self.total_batch_idx % self.hparams.train.log_interval == 0:
            examples = [
                wandb.Image(
                    utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().float().numpy()
                    ),
                    caption='target_mel'
                ),
                wandb.Image(utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().float().numpy()
                    ),
                    caption='gen_mel'
                ),
            ]

            self.logger.experiment.log({"mel_plots": examples})

        accumulate_grad_batches = self.hparams.train.get("accumulate_grad_batches", 1)
        should_update = (
            batch_idx + 1
        ) % accumulate_grad_batches == 0 or self.trainer.is_last_batch
        # optimizer
        self.manual_backward(loss_gen_all / accumulate_grad_batches)
        if should_update:
            self.log(
                "grad_norm_g", clip_grad_value_(self.net_g.parameters(), None)
            )
            optim_g.step()
            optim_g.zero_grad()
        self.untoggle_optimizer(optim_g)

        # Discriminator
        # train
        self.toggle_optimizer(optim_d)
        y_d_hat_r, y_d_hat_g, _, _ = self.net_d(wav, output_dict["model_outputs"].detach())

        # discriminator loss
        with autocast(enabled=False):
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc

        # log loss
        self.log("loss-d_total", loss_disc_all, prog_bar=True)

        # optimizer
        self.manual_backward(loss_disc_all / accumulate_grad_batches)
        if should_update:
            self.log(
                "grad-norm_d", clip_grad_value_(self.net_d.parameters(), None)
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
            x, x_lengths, y, y_lengths, wav, wav_lengths, sid, lid = batch
            y_mel = spec_to_mel_torch(
                y,
                self.hparams
            )
            y_hat = self.net_g.infer(x, x_lengths, sid, lid, y, y_lengths)
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1).float(), self.hparams)

            audio_examples = []
            audio_examples.append(wandb.Audio(wav[0][0].cpu().numpy(), caption='target_wav', sample_rate=self.hparams.data.sampling_rate))
            audio_examples.append(wandb.Audio(y_hat[0][0].cpu().numpy(), caption='reconstructed_wav', sample_rate=self.hparams.data.sampling_rate))
            self.logger.experiment.log({
                "audios_val": audio_examples
            })

            image_examples = [
                wandb.Image(
                    utils.plot_spectrogram_to_numpy(
                        y_mel[0].cpu().float().numpy()
                    ),
                    caption='target_mel'
                ),
                wandb.Image(
                    utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].cpu().float().numpy()
                    ),
                    caption='gen_mel'
                ),
            ]

            self.logger.experiment.log({"mel_plots_val": image_examples})

    def on_validation_end(self) -> None:
        self.save_checkpoints()