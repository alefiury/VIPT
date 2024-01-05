import os
import random
import time
import json

import numpy as np
import torch
import torchaudio
import torch.utils.data

from data.text.utils import symbols_to_ids
from data.audio.utils import load_wav_to_torch
from data.text.cleaners import english_cleaners2
from data.audio.mel_processing import (
    mel_spectrogram_torch,
    spectrogram_torch,
)

class TextAudioLoader(torch.utils.data.Dataset):
    """
    1) loads audio, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, metadata, hparams):
        self.hparams = hparams
        self.metadata = metadata
        self.text_cleaner = english_cleaners2
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate

        self.use_mel_spec_posterior = hparams.get("use_mel_posterior_encoder", False)
        if self.use_mel_spec_posterior:
            self.n_mel_channels = hparams.get("n_mel_channels", 80)
        self.cleaned_text = hparams.get("cleaned_text", False)

        self.add_blank = hparams.get("add_blank", False)
        self.min_text_len = hparams.get("min_text_len", 1)
        self.max_text_len = hparams.get("max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.metadata)

        self.speaker_dict = self.build_speaker_dict()
        self.save_speaker_dict(self.speaker_dict, hparams.speaker_dict_path)

        self.language_dict = self.build_language_dict()
        self.save_language_dict(self.language_dict, hparams.language_dict_path)

        self._filter()

    def build_speaker_dict(self):
        speaker_dict = {}
        for datum in self.metadata:
            speaker = datum["speaker_name"]
            if speaker not in speaker_dict:
                speaker_dict[speaker] = len(speaker_dict)
        return speaker_dict

    def save_speaker_dict(self, speaker_dict, save_path):
        with open(save_path, "w") as f:
            json.dump(speaker_dict, f, indent=2)

    def build_language_dict(self):
        language_dict = {}
        for datum in self.metadata:
            language = datum["language"]
            if language not in language_dict:
                language_dict[language] = len(language_dict)
        return language_dict

    def save_language_dict(self, language_dict, save_path):
        with open(save_path, "w") as f:
            json.dump(language_dict, f, indent=2)

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        metadata_new = []
        lengths = []
        for datum in self.metadata:
            text = datum["text"]
            audiopath = datum["audiopath"]
            language = datum["language"]
            speaker = datum["speaker_name"]

            language_id = self.language_dict[language]
            speaker_id = self.speaker_dict[speaker]

            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                metadata_new.append([audiopath, text, speaker_id, language_id])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.metadata = metadata_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text, speaker_id, language_id = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2], audiopath_and_text[3]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        return (text, spec, wav, speaker_id, language_id)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            # Resample if needed
            resample_transform = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
            audio = resample_transform(audio)
            sampling_rate = self.sampling_rate
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        return spec, audio_norm

    def get_text(self, text):
        phonemes = self.text_cleaner(text)
        phonemes_ids = symbols_to_ids(phonemes)
        phonemes_ids = torch.LongTensor(phonemes_ids)
        return phonemes_ids

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.metadata[index])

    def __len__(self):
        return len(self.metadata)


class TextAudioCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))
        lid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]
            lid[i] = row[4]

        if self.return_ids:
            return (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                sid,
                lid,
                ids_sorted_decreasing,
            )
        return (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            sid,
            lid,
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size