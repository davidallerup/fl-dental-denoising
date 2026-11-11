"""Train/val/test splits for Case Study 3B: half of every Case Study 3A client set.

See ``create_train_test_split_case3a.py`` for how the nested splits are built. Client 5
(Dataset 3) is halved as well, as in the runs that produced the published results.

Usage (from the repository root):
    python -m src.data_preparation.create_train_test_split_case3b
"""
from src.data_preparation.create_train_test_split_case3a import write_fragmented_splits
from src.paths import DATA_ROOT

if __name__ == "__main__":
    write_fragmented_splits(DATA_ROOT / "dental_train_val_test_splits_case3b", seed=42, keep_half=True)
