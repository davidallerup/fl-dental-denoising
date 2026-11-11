"""Local training: one model per client, trained only on that client's own data.

Usage (from the repository root):
    python -m other_training.local_training --experiment case1 --client 0

This uses the same training loop and hyperparameters as centralized training
(``other_training/centralized_training.py``); only the data differs. Checkpoints are
written to ``DATA_ROOT/torch_models/local_client<N>_<experiment>/local_model_epochNN.pth``.
"""

import argparse

from federated_learning_denoising_dentistry.config import EXPERIMENTS, client_memmaps
from other_training.centralized_training import EPOCHS, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="case1", choices=list(EXPERIMENTS))
    parser.add_argument("--client", type=int, required=True, help="Client index within the experiment")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    run_training(
        train_info=client_memmaps(args.experiment, args.client, "train"),
        val_info=client_memmaps(args.experiment, args.client, "val"),
        run_name=f"local_client{args.client}_{args.experiment}",
        checkpoint_prefix="local_model",
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
