import os
import re
import warnings
from itertools import groupby
from pathlib import Path
from typing import Any
from logging import getLogger

import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import torch.nn.functional as F
from torch import nn

LOG = getLogger(__name__)
IS_COLAB = os.getenv("COLAB_RELEASE_TAG", False)

def _substitute_if_same_shape(to_: dict[str, Any], from_: dict[str, Any]) -> None:
    not_in_to = list(filter(lambda x: x not in to_, from_.keys()))
    not_in_from = list(filter(lambda x: x not in from_, to_.keys()))
    if not_in_to:
        warnings.warn(f"Keys not found in model state dict:" f"{not_in_to}")
    if not_in_from:
        warnings.warn(f"Keys not found in checkpoint state dict:" f"{not_in_from}")
    shape_missmatch = []
    for k, v in from_.items():
        if k not in to_:
            pass
        elif hasattr(v, "shape"):
            if not hasattr(to_[k], "shape"):
                raise ValueError(f"Key {k} is not a tensor")
            if to_[k].shape == v.shape:
                to_[k] = v
            else:
                shape_missmatch.append((k, to_[k].shape, v.shape))
        elif isinstance(v, dict):
            assert isinstance(to_[k], dict)
            _substitute_if_same_shape(to_[k], v)
        else:
            to_[k] = v
    if shape_missmatch:
        warnings.warn(
            f"Shape mismatch: {[f'{k}: {v1} -> {v2}' for k, v1, v2 in shape_missmatch]}"
        )


def safe_load(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    model_state_dict = model.state_dict()
    _substitute_if_same_shape(model_state_dict, state_dict)
    model.load_state_dict(model_state_dict)


def load_checkpoint(
    checkpoint_path: Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    skip_optimizer: bool = False,
) -> tuple[torch.nn.Module, torch.optim.Optimizer | None, float, int]:
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"File {checkpoint_path} not found")
    with Path(checkpoint_path).open("rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="TypedStorage is deprecated"
            )
            checkpoint_dict = torch.load(f, map_location="cpu", weights_only=True)
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]

    # safe load module
    if hasattr(model, "module"):
        safe_load(model.module, checkpoint_dict["model"])
    else:
        safe_load(model, checkpoint_dict["model"])
    # safe load optim
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            safe_load(optimizer, checkpoint_dict["optimizer"])

    LOG.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {iteration})")
    return model, optimizer, learning_rate, iteration


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    iteration: int,
    checkpoint_path: Path | str,
) -> None:
    LOG.info(
        "Saving model and optimizer state at epoch {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    with Path(checkpoint_path).open("wb") as f:
        torch.save(
            {
                "model": state_dict,
                "iteration": iteration,
                "optimizer": optimizer.state_dict(),
                "learning_rate": learning_rate,
            },
            f,
        )


def clean_checkpoints(
    path_to_models: Path | str, n_ckpts_to_keep: int = 2, sort_by_time: bool = True
) -> None:
    """Freeing up space by deleting saved ckpts

    Arguments:
    path_to_models    --  Path to the model directory
    n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
    sort_by_time      --  True -> chronologically delete ckpts
                          False -> lexicographically delete ckpts
    """
    LOG.info("Cleaning old checkpoints...")
    path_to_models = Path(path_to_models)

    # Define sort key functions
    name_key = lambda p: int(re.match(r"[GD]_(\d+)", p.stem).group(1))
    time_key = lambda p: p.stat().st_mtime
    path_key = lambda p: (p.stem[0], time_key(p) if sort_by_time else name_key(p))

    models = list(
        filter(
            lambda p: (
                p.is_file()
                and re.match(r"[GD]_\d+", p.stem)
                and not p.stem.endswith("_0")
            ),
            path_to_models.glob("*.pth"),
        )
    )

    models_sorted = sorted(models, key=path_key)

    models_sorted_grouped = groupby(models_sorted, lambda p: p.stem[0])

    for group_name, group_items in models_sorted_grouped:
        to_delete_list = list(group_items)[:-n_ckpts_to_keep]

        for to_delete in to_delete_list:
            if to_delete.exists():
                LOG.info(f"Removing {to_delete}")
                if IS_COLAB:
                    to_delete.write_text("")
                to_delete.unlink()


def latest_checkpoint_path(dir_path: Path | str, regex: str = "G_*.pth") -> Path | None:
    dir_path = Path(dir_path)
    name_key = lambda p: int(re.match(r"._(\d+)\.pth", p.name).group(1))
    paths = list(sorted(dir_path.glob(regex), key=name_key))
    if len(paths) == 0:
        return None
    return paths[-1]


def plot_data_to_numpy(x: ndarray, y: ndarray) -> ndarray:
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(10, 2))
    plt.plot(x)
    plt.plot(y)
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram: ndarray) -> ndarray:
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data