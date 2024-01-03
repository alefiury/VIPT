import math
import warnings
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from modules.helpers import sequence_mask
from modules.commons import rand_slice_segments
from modules.hifigan_generator import HifiganGenerator
from modules.duration_predictor import DurationPredictor
from modules.modules import LengthRegulator, FramePriorNet
from modules.modules import TextEncoder, PosteriorEncoder, ResidualCouplingBlocks, ProsodyEncoder


class VIPT(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab: int,
        hidden_channels: int,
        hidden_channels_ffn_text_encoder: int,
        num_heads_text_encoder: int,
        num_layers_text_encoder: int,
        kernel_size_text_encoder: int,
        dropout_p_text_encoder: int,
        spec_channels: int,
        kernel_size_posterior_encoder: int,
        dilation_rate_posterior_encoder: int,
        num_layers_posterior_encoder: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: int,
        n_speakers: int,
        n_languages: int,
        embedded_speaker_dim: int,
        embedded_language_dim: int,
        prosody_emb_dim: int,
        kernel_size_flow: int,
        dilation_rate_flow: int,
        num_layers_flow: int,
        resblock_type_decoder: Literal["original", "simplified"],
        resblock_dilation_sizes_decoder: Sequence[int],
        resblock_kernel_sizes_decoder: Sequence[int],
        upsample_kernel_sizes_decoder: Sequence[int],
        upsample_initial_channel_decoder: int,
        upsample_rates_decoder: Sequence[int],
        segment_size: int,
        **kwargs: Any,
    ):
        super().__init__()

        self.segment_size = segment_size

        if kwargs:
            warnings.warn(f"Unused arguments: {kwargs}")

        self.emb_g = nn.Embedding(n_speakers, embedded_speaker_dim)
        self.emb_l = nn.Embedding(n_languages, embedded_language_dim)

        self.prosody_encoder = ProsodyEncoder(
            conv_filters=[spec_channels, 512, 512, 512, 512],
            lstm_units=512,
            z_dim=embedded_speaker_dim,
        )

        self.text_encoder = TextEncoder(
            n_vocab=n_vocab,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn_text_encoder,
            num_heads=num_heads_text_encoder,
            num_layers=num_layers_text_encoder,
            kernel_size=kernel_size_text_encoder,
            dropout_p=dropout_p_text_encoder,
            language_emb_dim=embedded_language_dim,
            prosody_emb_dim=prosody_emb_dim
        )

        self.posterior_encoder = PosteriorEncoder(
            in_channels=spec_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size_posterior_encoder,
            dilation_rate=dilation_rate_posterior_encoder,
            num_layers=num_layers_posterior_encoder,
            cond_channels=embedded_speaker_dim,
        )

        self.duration_predictor = DurationPredictor(
            in_channels=hidden_channels+embedded_language_dim+prosody_emb_dim,
            filter_channels=256,
            kernel_size=3,
            p_dropout=0.5,
            gin_channels=embedded_speaker_dim
        )

        self.lr = LengthRegulator()

        self.flow = ResidualCouplingBlocks(
            channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size_flow,
            dilation_rate=dilation_rate_flow,
            num_layers=num_layers_flow,
            cond_channels=embedded_speaker_dim,
        )

        self.frame_prior_net = FramePriorNet(
            n_vocab=n_vocab,
            out_channels=hidden_channels+embedded_language_dim+prosody_emb_dim,
            hidden_channels=hidden_channels+embedded_language_dim+prosody_emb_dim,
            filter_channels=hidden_channels_ffn_text_encoder,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
        )

        self.waveform_decoder = HifiganGenerator(
            in_channels=hidden_channels,
            out_channels=1,
            resblock_type=resblock_type_decoder,
            resblock_dilation_sizes=resblock_dilation_sizes_decoder,
            resblock_kernel_sizes=resblock_kernel_sizes_decoder,
            upsample_kernel_sizes=upsample_kernel_sizes_decoder,
            upsample_initial_channel=upsample_initial_channel_decoder,
            upsample_factors=upsample_rates_decoder,
            inference_padding=0,
            cond_channels=embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

    def reparameterize_vae(self, mu, logvar):
        """
        Applies the reparameterization trick: z = mu + sigma * epsilon,
        where epsilon is sampled from a standard normal distribution.
        """
        std = torch.exp(0.5 * logvar)  # standard deviation
        eps = torch.randn_like(std)  # epsilon ~ N(0, 1)
        return mu + eps * std

    def forward(
        self,
        x: torch.tensor,
        x_lengths: torch.tensor,
        sid: torch.tensor,
        lid: torch.tensor,
        y: torch.tensor,
        y_lengths: torch.tensor
    ) -> Dict:
        """Forward pass of the model.

        Args:
            x (torch.tensor): Batch of input character sequence IDs.
            x_lengths (torch.tensor): Batch of input character sequence lengths.
            y (torch.tensor): Batch of input spectrograms.
            y_lengths (torch.tensor): Batch of input spectrogram lengths.
            waveform (torch.tensor): Batch of ground truth waveforms per sample.
            aux_input (dict, optional): Auxiliary inputs for multi-speaker and multi-lingual training.
                Defaults to {"d_vectors": None, "speaker_ids": None, "language_ids": None}.

        Returns:
            Dict: model outputs keyed by the output name.

        Shapes:
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - sid: :math:`[B]`
            - lid: :math:`[B]`
            - y: :math:`[B, C, T_spec]`
            - y_lengths: :math:`[B]`

        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - ids_slice: :math:`[B, T_wav]`
            - x_mask: :math:`[B, 1, T_spec]`
            - y_mask: :math:`[B, 1, T_spec]`
            - m_p: :math:`[B, T_spec, C]`
            - logs_p: :math:`[B, T_spec, C]`
            - z: :math:`[B, T_spec, C]`
            - z_p: :math:`[B, T_spec, C]`
            - m_q: :math:`[B, T_spec, C]`
            - logs_q: :math:`[B, T_spec, C]`
        """
        outputs = {}
        # speaker embedding
        g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        # language embedding
        lang_emb = self.emb_l(lid).unsqueeze(-1)
        # prosody embedding
        pros_mu, pros_logvar = self.prosody_encoder(y)
        # reparameterize
        prosody_emb = self.reparameterize_vae(pros_mu, pros_logvar).unsqueeze(-1)
        # text encoder
        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb, prosody_emb=prosody_emb)
        # duration predictor
        log_duration_prediction = self.duration_predictor(x, x_mask, g=g)
        # duration predictor
        duration_rounded = torch.clamp(
            (torch.round(torch.exp(log_duration_prediction) - 1) * 1.0),
            min=0,
        ).long()

        x_frame, x_lengths = self.lr(x, duration_rounded, x_lengths)

        x_frame = x_frame.to(x.device)
        x_mask = torch.unsqueeze(
            sequence_mask(
                x_lengths,
                x_frame.size(2)
            ),
            1
        ).to(x.dtype)

        x_mask = x_mask.to(x.device)

        x_frame = self.frame_prior_net(x_frame, x_mask)

        # posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g)

        # flow layers
        z_p = self.flow(z, y_mask, g=g)

        z_slice, ids_slice = rand_slice_segments(
            z, y_lengths, self.segment_size
        )

        o = self.waveform_decoder(z_slice, g=g)

        outputs.update(
            {
                "model_outputs": o,
                "ids_slice": ids_slice,
                "x_mask": x_mask,
                "y_mask": y_mask,
                "m_p": m_p,
                "logs_p": logs_p,
                "z": z,
                "z_p": z_p,
                "m_q": m_q,
                "logs_q": logs_q,
            }
        )

        return outputs

    def infer(
        self,
        x: torch.tensor,
        x_lengths: torch.tensor,
        y: torch.tensor,
        y_lengths: torch.tensor,
        sid: torch.tensor,
        lid: torch.tensor,
    ):
        outputs = {}
        # speaker embedding
        g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        # language embedding
        lang_emb = self.emb_l(lid).unsqueeze(-1)
        # prosody embedding
        pros_mu, pros_logvar = self.prosody_encoder(y)
        # reparameterize
        prosody_emb = self.reparameterize_vae(pros_mu, pros_logvar)
        # text encoder
        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb, prosody_emb=prosody_emb)
        # duration predictor
        log_duration_prediction = self.duration_predictor(x, x_mask, g=g)
        # duration predictor
        duration_rounded = torch.clamp(
            (torch.round(torch.exp(log_duration_prediction) - 1) * 1.0),
            min=0,
        ).long()

        x_frame, x_lengths = self.lr(x, duration_rounded, x_lengths)

        x_frame = x_frame.to(x.device)
        x_mask = torch.unsqueeze(
            sequence_mask(
                x_lengths,
                x_frame.size(2)
            ),
            1
        ).to(x.dtype)

        x_mask = x_mask.to(x.device)

        z_p = self.frame_prior_net(x_frame, x_mask)

        # flow layers
        z = self.flow(z_p, x_mask, g=g, reverse=True)

        o = self.waveform_decoder(z, g=g)

        return o