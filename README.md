# Federated, Local and Centralized Learning for Dental Panoramic Radiograph Denoising

Code for the paper **"Impact of Data Heterogeneity on Federated, Local and Centralized Learning for Dental Panoramic Radiograph Denoising"** (Allerup, ur Rehman, Pauwels and Naveed).

We train a residual U-Net to denoise dental panoramic radiographs and compare three learning paradigms on multi-vendor, multi-institutional data:

- **Federated learning (FL):** clients train locally and share only model weights, aggregated with FedAvg using [Flower](https://flower.ai).
- **Local learning (LL):** each client trains a model on its own data only.
- **Centralized learning (CL):** one model trained on all clients' data pooled together.

Four case studies vary client imbalance, client fragmentation, local data availability and client-specific noise.

## Repository structure

```
federated_learning_denoising_dentistry/   Flower app (FL): client, server, data loading, experiment config
other_training/
  centralized_training.py                 Centralized learning (CL)
  local_training.py                       Local learning (LL), one model per client
src/
  data_preparation/                       Sorting raw datasets, train/val/test splits, noise, patch memmaps
  models/                                 Residual U-Net (used in the paper) and a plain U-Net
  paths.py                                DATA_ROOT / REPO_ROOT
notebooks-paper/
  caseN_inference*.ipynb                  Evaluate trained models on the test sets -> notebooks-paper/data/*.csv
  analysis-notebooks/                     Result tables in the paper, and the t-SNE figure
  data/                                   Per-image results (PSNR, MSE, SSIM, LPIPS, GMSD) behind every table
```

## Installation

```bash
git clone <this-repository-url>
cd <repository-folder>
curl -sSL https://install.python-poetry.org | python3 -   # Poetry 2.x
poetry config virtualenvs.in-project true                # optional
poetry install
```

Python 3.10. `pyproject.toml` installs PyTorch from the CUDA 11.8 wheel index; for CPU-only or Apple Silicon, remove `source = "pytorch-cu118"` from the `torch`/`torchvision`/`torchaudio` lines. For AMD GPUs, install the matching ROCm build of PyTorch instead.

Run all commands from the repository root. Training logs to [Weights & Biases](https://wandb.ai); run `wandb login` first, or `export WANDB_MODE=offline` to log locally only.

## Data

The four datasets are publicly available but not redistributed here. Download them from their sources and follow their terms of use.

| Dataset | Images | Subsets in the paper | Use |
|---|---|---|---|
| DENTEX ([Hugging Face](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)) | 3903 | Datasets 1, 2, 3 | Training clients |
| A Dual-Labeled Dataset (ADLD), Zhou et al. 2024 | 2066 | Datasets 4, 5 | Training clients |
| Tufts Dental Database ([tdd.ece.tufts.edu](https://tdd.ece.tufts.edu/)) | 1000 | Dataset 6 | Training client |
| USPFORP, Costa et al. 2024 | 936 | Dataset 7 | External test set only |

All data lives under one folder, `DATA_ROOT`. It defaults to `./data`; set `FL_DENTAL_DATA_ROOT` to use another location:

```bash
export FL_DENTAL_DATA_ROOT=/path/to/data
```

Layout after data preparation:

```
DATA_ROOT/
  raw/dentex/                             DENTEX download
  raw_tufts_jpg/, raw_uspforp_jpg/        Tufts / USPFORP downloads
  raw_sorted_dentex/client{0,1,2}/        DENTEX sorted into Datasets 1, 2, 3
  raw_sorted_dual_labeled/dl-client{1,2}/ ADLD sorted into Datasets 4, 5
  raw_tufts/, raw_uspforp/                Converted Tufts (Dataset 6) / USPFORP (Dataset 7)
  dental_train_val_test_splits_<case>/    {train,val,test}/clientN/ image splits (70/15/15, seed 42)
  memmap_<case>/                          Per-client 256x256 noisy/clean patches (FL and LL)
  memmap_centralized_<case>/              All clients combined (CL)
  torch_models/                           Trained models, written here by the training scripts
```

`<case>` is one of `case1`, `case2`, `case3a`, `case3b`, `case4`. Case Study 4 has no split folder of its own: it uses the `case1` splits with different noise.

### Preparation pipeline

The scripts in `src/data_preparation/` contain one block per case study in their `__main__` section; enable the block for the case study you want (the others are commented out), and create the output folders first. Run them as modules from the repository root, e.g. `python -m src.data_preparation.create_train_test_split`.

1. **Sort raw data into subsets:** `sort_dentex_data.py`, `sort_tufts_data.py`, `sort_uspforp_data.py`
2. **Train/val/test splits per client:** `create_train_test_split.py` (case1, case2), `create_train_test_split_case3a.py`, `create_train_test_split_case3b.py`
3. **Normalize to [0.1, 0.9], add noise and cut into 256×256 patches with 20 px overlap (memmaps):**
   - per client: `memmapped_train_val_test_data.py` (case1–case3b), `memmapped_train_val_test_data_diff_noise.py` (case4)
   - centralized: `centralized_memmapped_train_val_test_data.py` (case1, case2, case3b), `centralized_memmapped_train_val_test_data_diff_noise.py` (case4)

The noise models are in `src/data_preparation/add_noise.py` (Case Studies 1–3: quantum and electronic noise) and in `add_noise0`–`add_noise4` of the `*_diff_noise.py` scripts (Case Study 4, one model per client).

## Case studies

Experiments in the code are named after the paper's case studies. Paper datasets are numbered 1–7 as in the paper's dataset figure; paper client names for Case Studies 3A/3B are in brackets.

| Code | Paper | FL clients | Clients (code index → paper dataset) |
|---|---|---|---|
| `case1` | Case Study 1: balanced | 5 | 0 → D1 (half), 1 → D2, 2 → D4, 3 → D5, 4 → D6 |
| `case2` | Case Study 2: imbalanced | 6 | 0 → D1 (all), 1 → D2, 2 → D4, 3 → D5, 4 → D6, 5 → D3 |
| `case3a` | Case Study 3A: fragmented | 11 | Each Case Study 2 client except 5 split in two: 0/6 [0a/0b], 1/7 [1a/1b], 2/8 [2a/2b], 3/9 [3a/3b], 4/10 [4a/4b], 5 [5] |
| `case3b` | Case Study 3B: fragmented, half data | 11 | Half of each case3a client's train/val/test images, including client 5 |

The Case Study 3A/3B splits are nested in the Case Study 2 split: each Case Study 2 train, validation and test set is divided between the two clients (3A), and 3B keeps half of each 3A set. No image moves between training and testing across case studies, and the pooled 3A training data is exactly the Case Study 2 training data.
| `case4` | Case Study 4: heterogeneous noise | 5 | As case1, client *i* noised with `add_noise`*i* |

LL models were trained for clients 0–5 (0–4 in Case Studies 1 and 4). Models whose training data did not change between case studies were trained once and reused: Case Study 2 uses the `case1` LL models for clients 1–4, and Case Study 3A uses the `case2` CL model. The experiment definitions are in `EXPERIMENTS` in [`federated_learning_denoising_dentistry/config.py`](federated_learning_denoising_dentistry/config.py).

### Training

```bash
# Federated learning (one Flower federation per case study, sized to its number of clients)
flwr run . case1 --run-config "experiment='case1'"

# Centralized learning
python -m other_training.centralized_training --experiment case1

# Local learning, one run per client
python -m other_training.local_training --experiment case1 --client 0
```

Models are saved under `DATA_ROOT/torch_models/`:

| Paradigm | Saved as |
|---|---|
| FL | `federated_<case>/federated_model_sim_<timestamp>_round_NN.pth` |
| CL | `centralized_<case>/centralized_model_epochNN.pth` |
| LL | `local_client<N>_<case>/local_model_epochNN.pth` |

Settings that produced the paper's results:

| | FL | CL and LL |
|---|---|---|
| Model | Residual U-Net | Residual U-Net |
| Loss / optimizer | MSE, Adam, gradient-norm clipping at 1.0 | MSE, Adam, gradient-norm clipping at 1.0 |
| Batch size | 64 | 64 (Case Study 1 LL₀: 32) |
| Training length | 50 rounds × 1 local epoch, all clients every round, FedAvg | 50 epochs |
| Learning rate | 1e-4 for rounds 1–2, then 5e-5 | 1e-4, `ReduceLROnPlateau` on validation loss (factor 0.5, patience 5, relative threshold 1e-5, min 1e-6) |
| Evaluated model | Global model after round 50 | Checkpoint after epoch 50 |

Training was done on the LUMI supercomputer (one GCD of an AMD MI250X GPU); inference and evaluation on an NVIDIA RTX 4070 SUPER.

### Evaluation

Start Jupyter from the repository root with the root on the Python path, so the notebooks can import the project code:

```bash
PYTHONPATH=$(pwd) poetry run jupyter lab
```

The inference notebooks load models from the paths above, with placeholders you fill in for your own runs:

- `XX` in `epochXX` / `round_XX`: the checkpoint to evaluate (two digits, e.g. `epoch07`). Choose it from the validation losses in `LOGS/` or Weights & Biases, e.g. the epoch or round with the lowest validation loss.
- `XXXX` in federated model filenames: the run's start time (`YYYYMMDD_HHMMSS`), as written in the filename by your run.

1. `notebooks-paper/caseN_inference.ipynb` evaluates the FL, CL and LL models of a case study on the client test sets and writes per-image metrics to `notebooks-paper/data/caseN.csv`. Full radiographs are reconstructed from overlapping patch predictions, averaged in the overlaps. The `_uspforp` notebooks evaluate on the external USPFORP set; `case4_inference_uspforp_nm{0..4}` use each of the five Case Study 4 noise models.
2. `notebooks-paper/analysis-notebooks/` computes the paper's result tables from these CSVs: `inference_tables.ipynb` (means, standard deviations and 95% confidence intervals per model and case study, which are also the values plotted in the paper's error-bar figure) and `inference_tables_individual_testsets.ipynb` (results on individual client test sets). **The CSVs from our runs are included**, so all reported results can be regenerated without data or GPUs. `tsne_modalities.ipynb` produces the t-SNE figure from the raw datasets.

Test images are noised on the fly without a fixed seed, so re-running inference gives slightly different numbers; the included CSVs are the reference results.

## Citation

If you use this code, please cite:

```bibtex
@article{allerup_fl_dental_denoising,
  title   = {Impact of Data Heterogeneity on Federated, Local and Centralized Learning for Dental Panoramic Radiograph Denoising},
  author  = {Allerup, David Jonathan and ur Rehman, Naveed and Pauwels, Ruben and Naveed, Khuram},
  journal = {TODO},
  year    = {TODO}
}
```

## License

Code is released under the [Apache License 2.0](LICENSE). The datasets are subject to their own licenses.
