import os

import torch.nn as nn
import numpy as np
# Load yaml
from omegaconf import OmegaConf
import torch
from modules.model import VIPT
from data.ljspeech import ljspeech
from data.cleaners import english_cleaners2
from data.utils import symbols_to_ids, ids_to_symbols
from data.symbols import symbols_to_ids_dict

def main():
    # Load config
    config = OmegaConf.load("configs/default.yaml")

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

    print(metadata[0])

    print(english_cleaners2(metadata[0]["text"]))

    basic_symbols = symbols_to_ids(english_cleaners2(metadata[0]["text"]))

    print(basic_symbols)

    print(ids_to_symbols(basic_symbols))

if __name__ == '__main__':
    main()