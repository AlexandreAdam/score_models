import torch
from .utils import DEVICE, DTYPE
import h5py
iport numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, key, extension="h5", device=DEVICE):
        self.device = device
        self.key = key
        if extension in ["h5", "hdf5"]:
            self._data = h5py.File(self.filepath, "r")
            self.data = lambda index: self._data[self.key][[index]]
            self.size = self._data[self.key].shape[0]
        elif extension == "npy":
            self._data = np.load(path)
            self.data = lambda index: self._data[[index]]
            self.size = self.data.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.tensor(self.data(index), dtype=DTYPE).to(self.device)
