"""Shared settings and experiment definitions for federated, centralized and local training.

Memmaps are produced by the scripts in ``src/data_preparation`` and are named
``<split>_<clean|noisy>_client<N>_len<L>_size256x256.npy`` (per client) or
``combined_<split>_<clean|noisy>_len<L>_size256x256.npy`` (centralized). The number
of patches ``L`` is read from the filename, so nothing here needs editing per run.
"""
import re
from pathlib import Path
from typing import Dict

from src.paths import DATA_ROOT

DEBUGGING: bool = True
SAVE_SAMPLES: bool = True
WANDB_PROJECT_NAME: str = "rnd-federated-learning"
LOG_RUN_NAME: str = "sim"

MODEL_NAME: str = 'residual_unet' # residual_unet, unet

TRAIN_BATCH_SIZE: int = 64
VAL_BATCH_SIZE: int = 64

PATCH_SIZE: int = 256

# Experiments, named after the paper's Case Studies, and the data each one uses.
#   memmap_dir:             per-client memmaps (federated and local training), under DATA_ROOT
#   centralized_memmap_dir: all clients combined (centralized training); None if not trained separately
#   num_clients:            number of federated clients (= Flower SuperNodes)
# Case Study 3A pools the same data as Case Study 2, so it reuses the case2 CL model.
# Case Study 4 uses the Case Study 1 splits, with a different noise model per client.
EXPERIMENTS: Dict[str, Dict] = {
    "case1":  {"memmap_dir": "memmap_case1",  "centralized_memmap_dir": "memmap_centralized_case1",  "num_clients": 5},
    "case2":  {"memmap_dir": "memmap_case2",  "centralized_memmap_dir": "memmap_centralized_case2",  "num_clients": 6},
    "case3a": {"memmap_dir": "memmap_case3a", "centralized_memmap_dir": None,                        "num_clients": 11},
    "case3b": {"memmap_dir": "memmap_case3b", "centralized_memmap_dir": "memmap_centralized_case3b", "num_clients": 11},
    "case4":  {"memmap_dir": "memmap_case4",  "centralized_memmap_dir": "memmap_centralized_case4",  "num_clients": 5},
}
DEFAULT_EXPERIMENT: str = "case1"


def get_experiment(name: str) -> Dict:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{name}'. Choose one of: {', '.join(EXPERIMENTS)}")
    return EXPERIMENTS[name]


def _find_memmap_pair(directory: Path, name_template: str) -> Dict:
    """Locate a clean/noisy memmap pair and read its length from the filename.

    ``name_template`` contains ``{kind}``, which is filled with ``clean`` and ``noisy``.
    """
    files, lengths = {}, {}
    for kind in ("clean", "noisy"):
        name = name_template.format(kind=kind)
        pattern = re.compile(rf"^{re.escape(name)}_len(\d+)_size{PATCH_SIZE}x{PATCH_SIZE}\.npy$")
        matches = [(p, m) for p in (directory.iterdir() if directory.is_dir() else [])
                   if (m := pattern.match(p.name))]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one file matching '{name}_len<N>_size{PATCH_SIZE}x{PATCH_SIZE}.npy' "
                f"in {directory}, found {len(matches)}. Did you run the data preparation scripts?"
            )
        files[kind], lengths[kind] = matches[0][0], int(matches[0][1].group(1))
    if lengths["clean"] != lengths["noisy"]:
        raise ValueError(f"Clean and noisy memmaps in {directory} have different lengths: {lengths}")
    length = lengths["clean"]
    return {
        "clean": str(files["clean"]),
        "noisy": str(files["noisy"]),
        "len": length,
        "memmap_shape": (length, PATCH_SIZE, PATCH_SIZE),
    }


def client_memmaps(experiment: str, client_id: int, split: str) -> Dict:
    """Memmap paths and shape for one client's ``train`` or ``val`` split."""
    exp = get_experiment(experiment)
    if not 0 <= client_id < exp["num_clients"]:
        raise ValueError(f"{experiment} has clients 0..{exp['num_clients'] - 1}, got {client_id}")
    directory = DATA_ROOT / exp["memmap_dir"]
    return _find_memmap_pair(directory, f"{split}_{{kind}}_client{client_id}")


def centralized_memmaps(experiment: str, split: str) -> Dict:
    """Memmap paths and shape for the combined (all clients) ``train`` or ``val`` split."""
    exp = get_experiment(experiment)
    if exp["centralized_memmap_dir"] is None:
        raise ValueError(f"No separate centralized model for {experiment}: its pooled data equals case2's, "
                         f"so the paper reuses the case2 CL model (`--experiment case2`).")
    directory = DATA_ROOT / exp["centralized_memmap_dir"]
    return _find_memmap_pair(directory, f"combined_{split}_{{kind}}")
