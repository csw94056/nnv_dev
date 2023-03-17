import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import get_default_device
import time


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
class MnistModel(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers, out_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size

        self.frame_size = in_size
        self.frame_step = in_size

        dev = get_default_device()
        self.gru = nn.GRU(in_size, hidden_size, num_layers).to(dev)
        self.linear = nn.Linear(hidden_size, out_size).to(dev)

    def forward(self, xb):
        frames = xb.unfold(1, self.frame_size, self.frame_step)
        frames = frames.transpose(0, 1)
        _, (out,) = self.gru(frames)
        out = out[-1]
        out = self.linear(out)
        return out