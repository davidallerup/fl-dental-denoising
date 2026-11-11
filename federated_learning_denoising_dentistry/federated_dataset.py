import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from federated_learning_denoising_dentistry.config import client_memmaps


class MemmapPairDataset(Dataset):
    """Paired (noisy, clean) 256x256 patches stored in two float32 memmaps.

    ``memmap_info`` is a dict as returned by ``config.client_memmaps`` or
    ``config.centralized_memmaps`` (keys: ``clean``, ``noisy``, ``memmap_shape``).
    """

    def __init__(self, memmap_info: dict):
        self.clean_images = np.memmap(memmap_info["clean"], dtype=np.float32, mode="r",
                                      shape=memmap_info["memmap_shape"])
        self.noisy_images = np.memmap(memmap_info["noisy"], dtype=np.float32, mode="r",
                                      shape=memmap_info["memmap_shape"])

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean_img = torch.from_numpy(self.clean_images[idx].copy()).float()
        noisy_img = torch.from_numpy(self.noisy_images[idx].copy()).float()

        # Add channel dimension: (256, 256) -> (1, 256, 256)
        return noisy_img.unsqueeze(0), clean_img.unsqueeze(0)


class DentalDataset(MemmapPairDataset):
    """One federated client's train or val split for a given experiment."""

    def __init__(self, experiment: str, client_id: int, split: str = "train"):
        super().__init__(client_memmaps(experiment, client_id, "train" if split == "train" else "val"))


def make_dataloaders(train_dataset, val_dataset, train_batch_size, val_batch_size, num_workers=4):
    # No data augmentation is applied (the paper's results were produced without it).
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def get_dataloader(experiment, client_id, train_batch_size=16, val_batch_size=16, num_workers=4):
    return make_dataloaders(
        DentalDataset(experiment=experiment, client_id=client_id, split="train"),
        DentalDataset(experiment=experiment, client_id=client_id, split="val"),
        train_batch_size, val_batch_size, num_workers,
    )
