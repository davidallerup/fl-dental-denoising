"""Train/val/test splits for Case Study 3A (and, via ``keep_half``, Case Study 3B).

The splits are nested inside the Case Study 2 split, so no image changes role between
case studies:

1. Each dataset is split into train/val/test exactly as in Case Study 2
   (``split_data`` from ``create_train_test_split.py``, same seed).
2. Each of those three sets is halved into two clients (a/b).
3. Case Study 3B (``keep_half=True``) keeps half of every 3A client set.

As a result, the pooled Case Study 3A training data equals the Case Study 2 training data
(so the Case Study 2 CL model can be reused), Case Study 3A/3B test images are Case Study 2
test images, and 3B is a subset of 3A.

Usage (from the repository root):
    python -m src.data_preparation.create_train_test_split_case3a
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Union

from sklearn.model_selection import train_test_split

from src.data_preparation.create_train_test_split import copy_images, list_images, split_data
from src.paths import DATA_ROOT

SPLITS = ("train", "val", "test")

# (source folder under DATA_ROOT, client name(s)). Two names: the dataset is divided between
# two clients (a, b). One name: the client is kept whole (Dataset 3, client 5).
CLIENTS: List[tuple] = [
    ("raw_sorted_dentex/client0", ["client0", "client6"]),         # Dataset 1 -> 0a / 0b
    ("raw_sorted_dentex/client1", ["client1", "client7"]),         # Dataset 2 -> 1a / 1b
    ("raw_sorted_dual_labeled/dl-client1", ["client2", "client8"]),  # Dataset 4 -> 2a / 2b
    ("raw_sorted_dual_labeled/dl-client2", ["client3", "client9"]),  # Dataset 5 -> 3a / 3b
    ("raw_tufts", ["client4", "client10"]),                        # Dataset 6 -> 4a / 4b
    ("raw_sorted_dentex/client2", "client5"),                      # Dataset 3 -> 5
]


def _halve(images: Sequence[Path], seed: int):
    if len(images) < 2:
        return list(images), []
    first, second = train_test_split(list(images), test_size=0.5, random_state=seed, shuffle=True)
    return first, second


def fragmented_splits(source_root: Union[str, Path], client_names: Union[str, List[str]],
                      seed: int = 42, keep_half: bool = False) -> Dict[str, Dict[str, List[Path]]]:
    """Return ``{client_name: {"train": [...], "val": [...], "test": [...]}}`` without copying anything."""
    images = list_images(Path(source_root))
    if not images:
        raise RuntimeError(f"No images found in {source_root}")

    # Step 1: identical to the Case Study 2 split of this dataset.
    base = dict(zip(SPLITS, split_data(images, seed)))

    # Step 2: divide each set between the two clients.
    if isinstance(client_names, str):
        clients = {client_names: base}
    else:
        if len(client_names) != 2:
            raise ValueError("client_names must be one name or a list of two names")
        clients = {name: {} for name in client_names}
        for split, imgs in base.items():
            clients[client_names[0]][split], clients[client_names[1]][split] = _halve(imgs, seed)

    # Step 3 (Case Study 3B): keep half of every client set.
    if keep_half:
        for sets in clients.values():
            for split in SPLITS:
                sets[split], _ = _halve(sets[split], seed)
    return clients


def write_fragmented_splits(output_root: Union[str, Path], seed: int = 42, keep_half: bool = False,
                            dry_run: bool = False) -> None:
    output_base = Path(output_root)
    for source, names in CLIENTS:
        for name, sets in fragmented_splits(DATA_ROOT / source, names, seed, keep_half).items():
            print(f"{name}: " + ", ".join(f"{s}={len(sets[s])}" for s in SPLITS) + f"  (from {source})")
            for split in SPLITS:
                copy_images(sets[split], output_base / split / name, dry_run=dry_run)


if __name__ == "__main__":
    write_fragmented_splits(DATA_ROOT / "dental_train_val_test_splits_case3a", seed=42, keep_half=False)
