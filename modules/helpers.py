import os
from typing import Any, Callable, Dict, Union

from pathlib import Path
import sys
import fsspec
import torch

import torch
import numpy as np
from torch.nn import functional as F


def maximum_path(value, mask):
    return maximum_path_numpy(value, mask)


def maximum_path_numpy(value, mask, max_neg_val=None):
    """
    Monotonic alignment search algorithm
    Numpy-friendly version. It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    if max_neg_val is None:
        max_neg_val = -np.inf  # Patch for Sphinx complaint
    value = value * mask

    device = value.device
    dtype = value.dtype
    value = value.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy().astype(bool)

    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=max_neg_val)[:, :-1]
        v1 = v
        max_mask = v1 >= v0
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.astype(np.float32)
    path = torch.from_numpy(path).to(device=device, dtype=dtype)
    return path


def sequence_mask(sequence_length, max_len=None):
    """Create a sequence mask for filtering padding in a sequence tensor.

    Args:
        sequence_length (torch.tensor): Sequence lengths.
        max_len (int, Optional): Maximum sequence length. Defaults to None.

    Shapes:
        - mask: :math:`[B, T_max]`
    """
    if max_len is None:
        max_len = sequence_length.max()
    seq_range = torch.arange(max_len, dtype=sequence_length.dtype, device=sequence_length.device)
    # B x T_max
    return seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)


def segment(x: torch.tensor, segment_indices: torch.tensor, segment_size=4, pad_short=False):
    """Segment each sample in a batch based on the provided segment indices

    Args:
        x (torch.tensor): Input tensor.
        segment_indices (torch.tensor): Segment indices.
        segment_size (int): Expected output segment size.
        pad_short (bool): Pad the end of input tensor with zeros if shorter than the segment size.
    """
    # pad the input tensor if it is shorter than the segment size
    if pad_short and x.shape[-1] < segment_size:
        x = torch.nn.functional.pad(x, (0, segment_size - x.size(2)))

    segments = torch.zeros_like(x[:, :, :segment_size])

    for i in range(x.size(0)):
        index_start = segment_indices[i]
        index_end = index_start + segment_size
        x_i = x[i]
        if pad_short and index_end >= x.size(2):
            # pad the sample if it is shorter than the segment size
            x_i = torch.nn.functional.pad(x_i, (0, (index_end + 1) - x.size(2)))
        segments[i] = x_i[:, index_start:index_end]
    return segments


def rand_segments(
    x: torch.tensor, x_lengths: torch.tensor = None, segment_size=4, let_short_samples=False, pad_short=False
):
    """Create random segments based on the input lengths.

    Args:
        x (torch.tensor): Input tensor.
        x_lengths (torch.tensor): Input lengths.
        segment_size (int): Expected output segment size.
        let_short_samples (bool): Allow shorter samples than the segment size.
        pad_short (bool): Pad the end of input tensor with zeros if shorter than the segment size.

    Shapes:
        - x: :math:`[B, C, T]`
        - x_lengths: :math:`[B]`
    """
    _x_lenghts = x_lengths.clone()
    B, _, T = x.size()
    if pad_short:
        if T < segment_size:
            x = torch.nn.functional.pad(x, (0, segment_size - T))
            T = segment_size
    if _x_lenghts is None:
        _x_lenghts = T
    len_diff = _x_lenghts - segment_size
    if let_short_samples:
        _x_lenghts[len_diff < 0] = segment_size
        len_diff = _x_lenghts - segment_size
    else:
        assert all(
            len_diff > 0
        ), f" [!] At least one sample is shorter than the segment size ({segment_size}). \n {_x_lenghts}"
    segment_indices = (torch.rand([B]).type_as(x) * (len_diff + 1)).long()
    ret = segment(x, segment_indices, segment_size, pad_short=pad_short)
    return ret, segment_indices


def get_user_data_dir(appname):
    TTS_HOME = os.environ.get("TTS_HOME")
    XDG_DATA_HOME = os.environ.get("XDG_DATA_HOME")
    if TTS_HOME is not None:
        ans = Path(TTS_HOME).expanduser().resolve(strict=False)
    elif XDG_DATA_HOME is not None:
        ans = Path(XDG_DATA_HOME).expanduser().resolve(strict=False)
    elif sys.platform == "win32":
        import winreg  # pylint: disable=import-outside-toplevel

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
        )
        dir_, _ = winreg.QueryValueEx(key, "Local AppData")
        ans = Path(dir_).resolve(strict=False)
    elif sys.platform == "darwin":
        ans = Path("~/Library/Application Support/").expanduser()
    else:
        ans = Path.home().joinpath(".local/share")
    return ans.joinpath(appname)

def load_fsspec(
    path: str,
    map_location: Union[str, Callable, torch.device, Dict[Union[str, torch.device], Union[str, torch.device]]] = None,
    cache: bool = True,
    **kwargs,
) -> Any:
    """Like torch.load but can load from other locations (e.g. s3:// , gs://).

    Args:
        path: Any path or url supported by fsspec.
        map_location: torch.device or str.
        cache: If True, cache a remote file locally for subsequent calls. It is cached under `get_user_data_dir()/tts_cache`. Defaults to True.
        **kwargs: Keyword arguments forwarded to torch.load.

    Returns:
        Object stored in path.
    """
    is_local = os.path.isdir(path) or os.path.isfile(path)
    if cache and not is_local:
        with fsspec.open(
            f"filecache::{path}",
            filecache={"cache_storage": str(get_user_data_dir("tts_cache"))},
            mode="rb",
        ) as f:
            return torch.load(f, map_location=map_location, **kwargs)
    else:
        with fsspec.open(path, "rb") as f:
            return torch.load(f, map_location=map_location, **kwargs)