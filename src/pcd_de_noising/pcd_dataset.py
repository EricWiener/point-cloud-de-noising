from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

DATA_KEYS = ["distance_m_1", "intensity_1"]
LABEL_KEY = "labels_1"


class PCDDataset(Dataset):
    """HDF5 PyTorch Dataset to load distance, intensity, and labels from the PCD dataset.

    Input params:
        file_path: Path to the folder containing the dataset (1+ HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
    """

    def __init__(self, file_path, recursive):
        super().__init__()
        self.files = []

        p = Path(file_path)
        assert p.is_dir()

        self.files = sorted(p.glob("**/*.hdf5" if recursive else "*.hdf5"))
        if len(self.files) < 1:
            raise RuntimeError("No hdf5 files found")

    def __getitem__(self, index):
        with h5py.File(self.files[index], "r") as h5_file:
            data = [h5_file[key][()] for key in DATA_KEYS]

            # 32 x 400
            label = h5_file[LABEL_KEY][()]

        data = tuple(torch.from_numpy(data) for data in data)
        data = torch.stack(data)  # 2 x 32 x 400

        distance = data[0:1, :, :]  # 1 x 32 x 400
        reflectivity = data[1:, :, :]  # 1 x 32 x 400

        label = torch.from_numpy(label).long()  # 32 x 400

        assert (
            label.shape == distance.shape[1:]
        ), "Label shape does not match distance shape"
        assert (
            label.shape == reflectivity.shape[1:]
        ), "Label shape does not match reflectivity shape"

        return distance, reflectivity, label

    def __len__(self):
        return len(self.files)
