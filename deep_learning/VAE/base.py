"""
VAE inferface
"""
from .types_ import *
from torch import nn
from abc import abstractmethod

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

class OneDimensionalDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler()
        self.data_normalized = torch.Tensor(self.scaler.fit_transform(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data_normalized[index]