import math
import warnings
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from modules.helpers import maximum_path
from modules.helpers import sequence_mask, generate_path
from modules.commons import rand_slice_segments
from modules.hifigan_generator import HifiganGenerator
from modules.duration_predictor import DurationPredictor
from modules.modules import LengthRegulator, FramePriorNet, Projection
from modules.modules import TextEncoder, PosteriorEncoder, ResidualCouplingBlocks, ProsodyEncoder
from modules.stochastic_duration_predictor import StochasticDurationPredictor


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
        use_sdp: bool = True,
        detach_dp_input: bool = True,
        inference_noise_scale_dp: float = 1.0,
        length_scale: float = 1.0,
        inference_noise_scale: float = 0.667,
        encoder_sample_rate: int = None,
        max_inference_len: int = None,
        **kwargs: Any,
    ):
        super().__init__()

        self.use_sdp = use_sdp
        self.segment_size = segment_size
        self.detach_dp_input = detach_dp_input
        self.inference_noise_scale_dp = inference_noise_scale_dp
        self.length_scale = length_scale
        self.inference_noise_scale = inference_noise_scale
        self.encoder_sample_rate = encoder_sample_rate
        self.max_inference_len = max_inference_len

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

        if use_sdp:
            self.duration_predictor = StochasticDurationPredictor(
                hidden_channels,
                192,
                3,
                0.5,
                4,
                cond_channels=embedded_speaker_dim,
                language_emb_dim=embedded_language_dim,
                prosody_emb_dim=prosody_emb_dim,
            )
        else:
            self.duration_predictor = DurationPredictor(
                hidden_channels,
                256,
                3,
                0.5,
                cond_channels=embedded_speaker_dim,
                language_emb_dim=embedded_language_dim,
                prosody_emb_dim=prosody_emb_dim,
            )

        self.flow = ResidualCouplingBlocks(
            channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size_flow,
            dilation_rate=dilation_rate_flow,
            num_layers=num_layers_flow,
            cond_channels=embedded_speaker_dim,
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


    def forward_mas(self, outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g, lang_emb, prosody_emb):
        # find the alignment path
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        with torch.no_grad():
            o_scale = torch.exp(-2 * logs_p)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.einsum("klm, kln -> kmn", [o_scale, -0.5 * (z_p**2)])
            logp3 = torch.einsum("klm, kln -> kmn", [m_p * o_scale, z_p])
            logp4 = torch.sum(-0.5 * (m_p**2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp2 + logp3 + logp1 + logp4
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t, t']

        # duration predictor
        attn_durations = attn.sum(3)
        if self.use_sdp:
            loss_duration = self.duration_predictor(
                x.detach() if self.detach_dp_input else x,
                x_mask,
                attn_durations,
                g=g.detach() if self.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.detach_dp_input and lang_emb is not None else lang_emb,
                prosody_emb=prosody_emb.detach() if self.detach_dp_input and prosody_emb is not None else prosody_emb,
            )
            loss_duration = loss_duration / torch.sum(x_mask)
        else:
            attn_log_durations = torch.log(attn_durations + 1e-6) * x_mask
            log_durations = self.duration_predictor(
                x.detach() if self.detach_dp_input else x,
                x_mask,
                g=g.detach() if self.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.detach_dp_input and lang_emb is not None else lang_emb,
                prosody_emb=prosody_emb.detach() if self.detach_dp_input and prosody_emb is not None else prosody_emb,
            )
            loss_duration = torch.sum((log_durations - attn_log_durations) ** 2, [1, 2]) / torch.sum(x_mask)
        outputs["loss_duration"] = loss_duration
        return outputs, attn

    def upsampling_z(self, z, slice_ids=None, y_lengths=None, y_mask=None):
        spec_segment_size = self.segment_size
        if self.encoder_sample_rate:
            # recompute the slices and spec_segment_size if needed
            slice_ids = slice_ids * int(self.interpolate_factor) if slice_ids is not None else slice_ids
            spec_segment_size = spec_segment_size * int(self.interpolate_factor)
            # interpolate z if needed
            if self.interpolate_z:
                z = torch.nn.functional.interpolate(z, scale_factor=[self.interpolate_factor], mode="linear").squeeze(0)
                # recompute the mask if needed
                if y_lengths is not None and y_mask is not None:
                    y_mask = (
                        sequence_mask(y_lengths * self.interpolate_factor, None).to(y_mask.dtype).unsqueeze(1)
                    )  # [B, 1, T_dec_resampled]

        return z, spec_segment_size, slice_ids, y_mask


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

        # posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g)

        # flow layers
        z_p = self.flow(z, y_mask, g=g)

        # duration predictor
        outputs, attn = self.forward_mas(outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g=g, lang_emb=lang_emb, prosody_emb=prosody_emb)

        # expand prior
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        # # expand prior
        # m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        # logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # slice segments
        z_slice, ids_slice = rand_slice_segments(
            z, y_lengths, self.segment_size
        )

        # vocoder
        o = self.waveform_decoder(z_slice, g=g)

        outputs.update(
            {
                "model_outputs": o,
                "ids_slice": ids_slice,
                "x_mask": x_mask,
                "z_mask": y_mask,
                "m_p": m_p,
                "logs_p": logs_p,
                "z": z,
                "z_p": z_p,
                "m_q": m_q,
                "logs_q": logs_q,
                "pros_mu": pros_mu,
                "pros_logvar": pros_logvar,
            }
        )

        return outputs

    def infer(
        self,
        x: torch.tensor,
        x_lengths: torch.tensor,
        sid: torch.tensor,
        lid: torch.tensor,
        y: torch.tensor,
        y_lengths: torch.tensor
    ):
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

        if self.use_sdp:
            logw = self.duration_predictor(
                x,
                x_mask,
                g=g,
                reverse=True,
                noise_scale=self.inference_noise_scale_dp,
                lang_emb=lang_emb,
                prosody_emb=prosody_emb,
            )
        else:
            logw = self.duration_predictor(
                x,
                x_mask,
                g=g,
                lang_emb=lang_emb,
                prosody_emb=prosody_emb,
            )

        w = torch.exp(logw) * x_mask * self.length_scale

        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)  # [B, 1, T_dec]

        attn_mask = x_mask * y_mask.transpose(1, 2)  # [B, 1, T_enc] * [B, T_dec, 1]
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)

        # upsampling if needed
        z, _, _, y_mask = self.upsampling_z(z, y_lengths=y_lengths, y_mask=y_mask)

        o = self.waveform_decoder((z * y_mask)[:, :, : self.max_inference_len], g=g)

        return o