"""federated-learning-denoising-dentistry: A Flower / PyTorch app."""

import torch
import os
import json
import csv

from random import random
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigsRecord
from federated_learning_denoising_dentistry.task import get_weights, load_data, set_weights, test, train

from federated_learning_denoising_dentistry.config import (
    MODEL_NAME, DEBUGGING, WANDB_PROJECT_NAME, DEFAULT_EXPERIMENT, get_experiment,
)
from src.models.model_factory import get_model

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, context: Context, partition_id):
        self.client_state = context.state
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.partition_id = partition_id

        if "fit_metrics" not in self.client_state.configs_records:
            self.client_state.configs_records["fit_metrics"] = ConfigsRecord()

        # Debugging to check if cuda is available as device
        # print(self.device)

    def fit(self, parameters, config):

        # Initialize local client model from global model.
        set_weights(self.net, parameters)

        if DEBUGGING:
            print("DEBUGGING: Client Fitting")

        train_dict, batch_stats = train(
            net=self.net,
            trainloader=self.trainloader,
            epochs=self.local_epochs,
            lr=config["lr"],
            device=self.device,
            partition_id=self.partition_id,
            project_name=WANDB_PROJECT_NAME
        )

        # get loss value from last epoch in this round
        train_loss, _ = train_dict.get(self.local_epochs - 1, None)

        # dump losses to log_file
        server_round = config.get("server_round")
        run_name = config.get("run_name")

        log_dir = f"LOGS/{run_name}"
        file_name = f"client{self.partition_id}_train_loss.csv"

        self._log_train_losses(log_dir=log_dir,
                               file_name=file_name,
                               server_round=server_round,
                               loss_dict=train_dict)
        
        self._log_batch_stats(log_dir=log_dir,
                              file_name=f"client{self.partition_id}_batch_stats.csv",
                              server_round=server_round,
                              batch_stats=batch_stats)

        return (
            get_weights(self.net),
            len(self.trainloader.dataset), # Training examples. Needed for weighted average aggregation
            {"train_loss": train_loss}, #, "my_metric": complex_metric_str
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)

        if DEBUGGING:
            print('CONFIG:\n', config)

        val_loss, val_psnr = test(
            net=self.net,
            testloader=self.valloader,
            device=self.device,
            partition_id=self.partition_id,
            server_round=config.get("server_round"),
            project_name=WANDB_PROJECT_NAME)
        
        # dump loss to log_file
        server_round = config.get("server_round")
        run_name = config.get("run_name")

        log_dir = f"LOGS/{run_name}"
        file_name = f"client{self.partition_id}_val_loss.csv"

        self._log_val_loss(log_dir=log_dir,
                               file_name=file_name,
                               server_round=server_round,
                               val_loss=val_loss,
                               val_psnr=val_psnr)

        return val_loss, len(self.valloader.dataset), {} # Last dict is a metrics dict (I think.)
    
    def _log_train_losses(self, log_dir: str, file_name: str, server_round: int, loss_dict: dict) -> None:

        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, file_name)

        file_exists = os.path.isfile(file_path)
        write_header = not file_exists or os.stat(file_path).st_size == 0

        with open(file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["round", "epoch", "loss", "psnr"])
            for epoch, tup in loss_dict.items():
                train_loss, train_psnr = tup
                writer.writerow([server_round, epoch, train_loss, train_psnr])

    def _log_batch_stats(self, log_dir: str, file_name: str, server_round: int, batch_stats: dict) -> None:

        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, file_name)

        file_exists = os.path.isfile(file_path)
        write_header = not file_exists or os.stat(file_path).st_size == 0

        with open(file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["round", "epoch", "batch", "loss", "psnr"])
            for epoch, batch_dict in batch_stats.items():
                for batch_idx, tup in batch_dict.items():
                    batch_loss, batch_psnr = tup
                    writer.writerow([server_round, epoch, batch_idx, batch_loss, batch_psnr])

    def _log_val_loss(self, log_dir: str, file_name: str, server_round: int, val_loss, val_psnr) -> None:

        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, file_name)

        file_exists = os.path.isfile(file_path)
        write_header = not file_exists or os.stat(file_path).st_size == 0

        with open(file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["round", "loss", "psnr"])

            writer.writerow([server_round, val_loss, val_psnr])




def client_fn(context: Context):
    # Load model and data
    net = get_model(MODEL_NAME)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    experiment = context.run_config.get("experiment", DEFAULT_EXPERIMENT)
    expected_clients = get_experiment(experiment)["num_clients"]
    if num_partitions != expected_clients:
        raise ValueError(
            f"{experiment} has {expected_clients} clients but the federation has {num_partitions} SuperNodes. "
            f"Run it with the matching federation, e.g. `flwr run . {experiment} --run-config \"experiment='{experiment}'\"`."
        )

    trainloader, valloader = load_data(partition_id, num_partitions, experiment)

    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net=net,
                        trainloader=trainloader,
                        valloader=valloader,
                        local_epochs=local_epochs,
                        context=context, partition_id=partition_id
                        ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
