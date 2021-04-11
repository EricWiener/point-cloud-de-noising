from pathlib import Path

import h5py
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

DATA_KEYS = ["distance_m_1", "intensity_1"]
LABEL_KEY = "labels_1"


class PCDDataset(Dataset):
    """HDF5 PyTorch Dataset to load distance, reflectivity, and labels from the PCD dataset.

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
            label = h5_file[LABEL_KEY][()]  # 32 x 400

        data = tuple(torch.from_numpy(data) for data in data)
        data = torch.stack(data)  # 2 x 32 x 400

        distance = data[0:1, :, :]  # 1 x 32 x 400
        reflectivity = data[1:, :, :]  # 1 x 32 x 400

        label = torch.from_numpy(label).long()  # 32 x 400
        # From the author: "We used the following label mapping for a point:
        #   0: no label, 100: valid/clear, 101: rain, 102: fog"
        # We will map these labels to the range [0, 3], where:
        #   0: no label, 1: valid/clear, 2: rain, 3: fog
        # TODO: Discard 0s? Not going to learn anything useful from them
        #   Might teach the model that adverse weather isn't adverse weather,
        #   because it's labeled as nothing
        label = torch.where(label == 0, torch.tensor(99), label)
        label -= 99

        assert (
            label.shape == distance.shape[1:]
        ), "Label shape does not match distance shape"
        assert (
            label.shape == reflectivity.shape[1:]
        ), "Label shape does not match reflectivity shape"

        return distance, reflectivity, label

    def __len__(self):
        return len(self.files)


class PointCloudDataModule(LightningDataModule):
    def __init__(self, train_directory, val_directory):
        """Create a PointCloudDataModule

        Args:
            train_directory (str): path to the training hdf5 files
            val_directory (str): path to the validation hdf5 files
        """
        super().__init__()
        self.train_directory = train_directory
        self.val_directory = val_directory

    def train_dataloader(self):
        dataset = PCDDataset(self.train_directory, recursive=True)
        print(f"Train found {len(dataset)} files")

        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
        return loader

    def val_dataloader(self):
        dataset = PCDDataset(self.val_directory, recursive=True)
        print(f"Val found {len(dataset)} files")

        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
        return loader
