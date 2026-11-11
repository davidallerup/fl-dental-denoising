"""federated-learning-denoising-dentistry: A Flower / PyTorch app."""

from collections import OrderedDict

import wandb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#from flwr_datasets import FederatedDataset
#from flwr_datasets.partitioner import DirichletPartitioner
#from torch.utils.data import DataLoader
#from torchvision.transforms import Compose, Normalize, ToTensor

from federated_learning_denoising_dentistry.federated_dataset import get_dataloader
from federated_learning_denoising_dentistry.config import DEBUGGING, SAVE_SAMPLES, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE

train_run_ids = {}
val_run_ids = {}

def load_data(partition_id: int, num_partitions: int, experiment: str):

    trainloader, testloader = get_dataloader(experiment=experiment,
                                             client_id=partition_id,
                                             train_batch_size=TRAIN_BATCH_SIZE,
                                             val_batch_size=VAL_BATCH_SIZE)

    return trainloader, testloader

def psnr(img1, img2):
    
    assert img1.shape == img2.shape, "Images must have the same dimensions"
    # Compute the Mean Squared Error
    #mse = np.mean((img1 - img2) ** 2)
    mse = F.mse_loss(img1, img2, reduction='mean').item()
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming the images are normalized between 0 and 1
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def batch_psnr(imgs1, imgs2):
    psnr_values = [psnr(imgs1[i], imgs2[i]) for i in range(imgs1.size(0))]
    return np.mean(psnr_values)

def train(net, trainloader, epochs, lr, device, partition_id, project_name: str):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Resume previous run if exists.
    run_id = train_run_ids.get(partition_id, None)
    
    if run_id is not None:
        # Resume the run if the ID exists
        wandb.init(project=project_name, name=f"Client-{str(partition_id)}-Train", id=run_id, resume="allow", settings=wandb.Settings(start_method="thread"))
        print(f"Resuming run with ID: {run_id}")

    else:
        # Create a new run if no ID exists for this client
        run = wandb.init(project=project_name, name=f"Client-{str(partition_id)}-Train", resume="allow")
        run_id = run.id  # Get the automatically generated ID
        train_run_ids[partition_id] = run_id  # Store the run ID for future rounds
        print(f"Created new run with ID: {run_id}")

    if DEBUGGING:
        print("DEBUGGING: Training")

    net.train()
    loss_dict = {}
    batch_stats = {}

    for epoch in range(epochs):

        batch_stats_tmp = {}

        running_loss = 0.0
        running_psnr = 0.0

        total_batches = len(trainloader)
        for batch_idx, (noisy_images, clean_images) in enumerate(trainloader):

            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            optimizer.zero_grad()
            outputs = net(noisy_images)

            loss = criterion(outputs, clean_images)

            loss.backward()

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()

            if DEBUGGING and (batch_idx) % 100 == 0:
                print(f"DEBUGGING - Training - Client {partition_id}: Batch {batch_idx+1} / {total_batches}")
                print(f"DEBUGGING - Client {partition_id}: Loss: ", loss.item())

                wandb.log({"batch_loss (every 100th batch)": loss.item()})
                wandb.log({"batch_psnr (every 100th batch)": batch_psnr(outputs, clean_images)})

                batch_stats_tmp[batch_idx] = (loss.item(), batch_psnr(outputs, clean_images))

            running_loss += loss.item()

            running_psnr += batch_psnr(outputs, clean_images)


        # Save loss for each of the epochs in this round
        train_loss = running_loss / len(trainloader)
        train_psnr = running_psnr / len(trainloader)
        loss_dict[epoch] = (train_loss, train_psnr)

        wandb.log({"train_loss": train_loss})
        wandb.log({"train_psnr": train_psnr})

        batch_stats[epoch] = batch_stats_tmp

    wandb.finish()

    # return last epoch loss
    return loss_dict, batch_stats


def test(net, testloader, device, partition_id, server_round, project_name: str):
    """Validate the model on the validation set."""
    criterion = nn.MSELoss()
    loss = 0.0
    psnr_value = 0.0
    net.eval()

    # Resume previous run if exists.
    run_id = val_run_ids.get(partition_id, None)
    
    if run_id is not None:
        # Resume the run if the ID exists
        wandb.init(project=project_name, name=f"Client-{str(partition_id)}-Val", id=run_id, resume="allow", settings=wandb.Settings(start_method="thread"))
        print(f"Resuming run with ID: {run_id}")

    else:
        # Create a new run if no ID exists for this client
        run = wandb.init(project=project_name, name=f"Client-{str(partition_id)}-Val", resume="allow")
        run_id = run.id  # Get the automatically generated ID
        val_run_ids[partition_id] = run_id  # Store the run ID for future rounds
        print(f"Created new run with ID: {run_id}")

    if DEBUGGING:
        print(f"DEBUGGING - Testing - Client {partition_id}")

    # Track saved samples
    samples_saved = 0
    max_samples = 40

    with torch.no_grad():
        for batch_idx, (noisy_images, clean_images) in enumerate(testloader):
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            outputs = net(noisy_images)
            loss += criterion(outputs, clean_images).item()
            psnr_value += batch_psnr(outputs, clean_images)

            # Save the first image from each batch up to 40 images
            if samples_saved < max_samples and noisy_images.shape[0] > 0:
                save_sample(
                    clean_img=clean_images[0],
                    noisy_img=noisy_images[0],
                    denoised_img=outputs[0],
                    filename=f"r{server_round}_client{partition_id}_val_sample_{samples_saved:02d}.png",
                    title=f"Client {partition_id} Validation Sample {samples_saved + 1}",
                    output_dir=f"SAMPLES/{server_round}"
                )
                samples_saved += 1
                
                #if samples_saved >= max_samples:
                #    print(f"Saved {max_samples} samples for Client {partition_id}")
                #    break  # Stop processing batches once we have 40 samples

    val_loss = loss / len(testloader)
    val_psnr = psnr_value / len(testloader)

    wandb.log({"val_loss": val_loss})
    wandb.log({"val_psnr": val_psnr})
    wandb.finish()

    return (val_loss, val_psnr)


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def save_sample(clean_img, noisy_img, denoised_img, filename, title=None, output_dir="SAMPLES"):
    import matplotlib.pyplot as plt
    import os

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
