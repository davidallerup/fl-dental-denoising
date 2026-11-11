"""Centralized training: one model trained on the combined data of all clients.

Usage (from the repository root):
    python -m other_training.centralized_training --experiment case1

Checkpoints are written to ``DATA_ROOT/torch_models/centralized_<experiment>/`` and per-epoch
losses to ``LOGS/centralized_<experiment>/``. The same training loop is used for the
local models, see ``other_training/local_training.py``.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from federated_learning_denoising_dentistry.config import EXPERIMENTS, MODEL_NAME, centralized_memmaps
from federated_learning_denoising_dentistry.federated_dataset import MemmapPairDataset, make_dataloaders
from src.models.model_factory import get_model
from src.paths import DATA_ROOT

# Settings used for the paper. The reported centralized and local models are the
# checkpoints after epoch 50 (the original runs continued to epoch 60; because the
# learning-rate schedule only reacts to past epochs, stopping at 50 gives the same model).
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCHS = 50
WANDB_PROJECT = "centralized-denoising-training"


def psnr(img1, img2):
    """Calculate PSNR between two images."""
    assert img1.shape == img2.shape, "Images must have the same dimensions"
    mse = F.mse_loss(img1, img2, reduction='mean').item()
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming images normalized between 0 and 1
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def batch_psnr(imgs1, imgs2):
    """Calculate average PSNR for a batch."""
    psnr_values = [psnr(imgs1[i], imgs2[i]) for i in range(imgs1.size(0))]
    return np.mean(psnr_values)


def save_sample(clean_img, noisy_img, denoised_img, filename, title=None, output_dir="SAMPLES"):
    """Save comparison image."""
    # Move tensors to CPU if needed
    clean_img = clean_img.cpu() if clean_img.is_cuda else clean_img
    noisy_img = noisy_img.cpu() if noisy_img.is_cuda else noisy_img
    denoised_img = denoised_img.cpu() if denoised_img.is_cuda else denoised_img

    def tensor_to_np(img):
        np_img = img.permute(1, 2, 0).numpy()
        if np_img.shape[-1] == 1:
            np_img = np_img.squeeze(-1)
        return np_img

    clean_np = tensor_to_np(clean_img)
    noisy_np = tensor_to_np(noisy_img)
    denoised_np = tensor_to_np(denoised_img)

    cmap = 'gray' if clean_np.ndim == 2 else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(clean_np, cmap=cmap)
    axes[0].set_title('Clean Image')
    axes[0].axis('off')

    axes[1].imshow(noisy_np, cmap=cmap)
    axes[1].set_title('Noisy Image')
    axes[1].axis('off')

    axes[2].imshow(denoised_np, cmap=cmap)
    axes[2].set_title('Denoised Image')
    axes[2].axis('off')

    if title:
        fig.suptitle(title)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison image to {save_path}")


def train_epoch(model, trainloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_psnr = 0.0

    for batch_idx, (noisy_images, clean_images) in enumerate(trainloader):
        noisy_images = noisy_images.to(device)
        clean_images = clean_images.to(device)

        optimizer.zero_grad()
        outputs = model(noisy_images)
        loss = criterion(outputs, clean_images)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        running_psnr += batch_psnr(outputs, clean_images)

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(trainloader)}, '
                  f'Loss: {loss.item():.6f}, PSNR: {batch_psnr(outputs, clean_images):.2f}')

    avg_loss = running_loss / len(trainloader)
    avg_psnr = running_psnr / len(trainloader)

    return avg_loss, avg_psnr


def validate(model, valloader, criterion, device, epoch, samples_dir, save_samples=True):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    samples_saved = 0
    max_samples = 20

    with torch.no_grad():
        for batch_idx, (noisy_images, clean_images) in enumerate(valloader):
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            outputs = model(noisy_images)
            loss = criterion(outputs, clean_images)

            val_loss += loss.item()
            val_psnr += batch_psnr(outputs, clean_images)

            # Save samples
            if save_samples and samples_saved < max_samples and noisy_images.shape[0] > 0:
                save_sample(
                    clean_img=clean_images[0],
                    noisy_img=noisy_images[0],
                    denoised_img=outputs[0],
                    filename=f"epoch{epoch:02d}_val_sample_{samples_saved:02d}.png",
                    title=f"Validation Sample {samples_saved + 1} - Epoch {epoch}",
                    output_dir=samples_dir
                )
                samples_saved += 1

    avg_loss = val_loss / len(valloader)
    avg_psnr = val_psnr / len(valloader)

    return avg_loss, avg_psnr


def _append_csv(path, header, row):
    write_header = not os.path.isfile(path) or os.stat(path).st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def run_training(train_info: dict, val_info: dict, run_name: str, checkpoint_prefix: str,
                 epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, learning_rate: float = LEARNING_RATE):
    """Train a denoising model on one pair of train/val memmaps.

    ``train_info`` / ``val_info`` come from ``config.centralized_memmaps`` or ``config.client_memmaps``.
    Checkpoints go to ``DATA_ROOT/torch_models/<run_name>/<checkpoint_prefix>_epochNN.pth``.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = os.path.join(DATA_ROOT, "torch_models", run_name)
    log_dir = os.path.join("LOGS", run_name)
    samples_dir = os.path.join("SAMPLES", run_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Train patches: {train_info['len']}, val patches: {val_info['len']}")

    wandb.init(project=WANDB_PROJECT, name=run_name)

    trainloader, valloader = make_dataloaders(
        MemmapPairDataset(train_info), MemmapPairDataset(val_info),
        train_batch_size=batch_size, val_batch_size=batch_size, num_workers=4,
    )

    model = get_model(MODEL_NAME)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=0.000001, threshold=0.00001)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_psnr = 0

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        train_loss, train_psnr = train_epoch(model, trainloader, criterion, optimizer, device, epoch)
        val_loss, val_psnr = validate(model, valloader, criterion, device, epoch, samples_dir)

        # Learning rate scheduling
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_psnr": train_psnr,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "learning_rate": lr,
        })
        _append_csv(os.path.join(log_dir, "train_losses.csv"), ["epoch", "train_loss", "train_psnr"],
                    [epoch, train_loss, train_psnr])
        _append_csv(os.path.join(log_dir, "val_losses.csv"), ["epoch", "val_loss", "val_psnr", "learning_rate"],
                    [epoch, val_loss, val_psnr, lr])

        print(f"Train Loss: {train_loss:.6f}, Train PSNR: {train_psnr:.2f}")
        print(f"Val Loss: {val_loss:.6f}, Val PSNR: {val_psnr:.2f}")
        print(f"Learning Rate: {lr:.6f}")

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            print(f"New best validation PSNR: {val_psnr:.2f} (epoch {epoch})")

        # Save a checkpoint every epoch (models are small); the paper uses epoch 50.
        epoch_model_path = os.path.join(save_dir, f"{checkpoint_prefix}_epoch{epoch:02d}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Model checkpoint saved: epoch {epoch} -> {epoch_model_path}")

    print(f"\nTraining complete! Best validation PSNR: {best_val_psnr:.2f}")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="case1", choices=list(EXPERIMENTS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    run_training(
        train_info=centralized_memmaps(args.experiment, "train"),
        val_info=centralized_memmaps(args.experiment, "val"),
        run_name=f"centralized_{args.experiment}",
        checkpoint_prefix="centralized_model",
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
