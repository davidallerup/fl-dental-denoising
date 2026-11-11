"""federated-learning-denoising-dentistry: A Flower / PyTorch app."""
import datetime
import json
import torch
import os

from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import logging

from federated_learning_denoising_dentistry.task import get_weights, set_weights
#from federated_learning_denoising_dentistry.custom_fed_avg import CustomFedAvg

from federated_learning_denoising_dentistry.config import DEBUGGING, MODEL_NAME, LOG_RUN_NAME
from src.models.model_factory import get_model
from src.paths import DATA_ROOT

# create a global run based on timestamp
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{LOG_RUN_NAME}_{run_timestamp}"

class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy that saves the model after each round."""
    
    def __init__(self, save_dir="torch_models", model_name=MODEL_NAME, run_name=run_name,
                 lr_factor=0.5, lr_patience=5, min_lr=0.000001,**kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir
        self.model_name = model_name
        self.run_name = run_name

        # ReduceLROnPlateau parameters
        self.current_lr = 0.0001
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.min_lr = min_lr

        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Model saving enabled. Models will be saved to: {self.save_dir}")
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results and save model."""
        # Call parent method to get aggregated parameters
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Calculate average loss from clients
        if results:
            total_examples = sum([fit_res.num_examples for _, fit_res in results])
            weighted_loss = sum([fit_res.num_examples * fit_res.metrics.get("train_loss", 0) 
                            for _, fit_res in results]) / total_examples
            
            # Update learning rate based on loss improvement
            self._update_learning_rate(weighted_loss)
            
            print(f"Round {server_round}: Avg Loss = {weighted_loss:.6f}, LR = {self.current_lr:.6f}")
        
        # Save model after aggregation
        if aggregated_parameters is not None:
            self._save_model(aggregated_parameters, server_round)
        
        return aggregated_parameters, aggregated_metrics
    
    def _update_learning_rate(self, current_loss):
        """Update learning rate using ReduceLROnPlateau logic."""
        improvement_threshold = 0.00001  # Minimum relative improvement
        
        # Check if there's significant improvement
        if current_loss < self.best_loss - (self.best_loss * improvement_threshold):
            self.best_loss = current_loss
            self.patience_counter = 0
            print(f"Loss improved to {current_loss:.6f}")
        else:
            self.patience_counter += 1
            print(f"No improvement. Patience: {self.patience_counter}/{self.lr_patience}")
            
            # Reduce learning rate if patience exceeded
            if self.patience_counter >= self.lr_patience:
                old_lr = self.current_lr
                self.current_lr = max(self.current_lr * self.lr_factor, self.min_lr)
                self.patience_counter = 0  # Reset patience
                print(f"Reducing learning rate from {old_lr:.6f} to {self.current_lr:.6f}")

    def get_current_lr(self):
        """Get current learning rate for on_fit_config."""
        return self.current_lr
    
    def _save_model(self, parameters: Parameters, round_num: int):
        """Save the aggregated model to disk."""
        try:
            # Create model instance
            net = get_model(self.model_name)
            
            # Convert Parameters object to ndarrays first
            parameters_ndarrays = parameters_to_ndarrays(parameters)
            
            # Set the aggregated weights
            set_weights(net, parameters_ndarrays)
            
            # Create filename
            model_filename = f"federated_model_{self.run_name}_round_{round_num:02d}.pth"
            model_path = os.path.join(self.save_dir, model_filename)
            
            # Save model state dict
            torch.save(net.state_dict(), model_path)
            print(f"Model saved: {model_path}")
            
            # Also save the latest model (overwrites each round)
            latest_path = os.path.join(self.save_dir, f"latest_model_{self.run_name}.pth")
            torch.save(net.state_dict(), latest_path)
            print(f"Latest model saved: {latest_path}")
            
        except Exception as e:
            print(f"Error saving model at round {round_num}: {e}")
            import traceback
            traceback.print_exc()

# Global strategy instance to access learning rate
strategy_instance = None

# Parameters needed for client fit and client evaluate should be added here :)
def on_fit_config(server_round: int) -> Metrics:

    # Get learning rate from strategy if available
    if strategy_instance and hasattr(strategy_instance, 'get_current_lr'):
        lr = strategy_instance.get_current_lr()
    else:
        # Fallback schedule
        lr = 0.0001 if server_round <= 2 else 0.00005

    return {
        "lr": lr,
        "run_name": run_name,
        "server_round": server_round
        }

def on_evaluate_config(server_round: int) -> Metrics:
    """Return evaluation configuration dict for each round."""
    return {
        "server_round": server_round,
        "evaluation_config": True,
        "run_name": run_name
    }

def server_fn(context: Context):
    global strategy_instance

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    experiment = context.run_config.get("experiment", "case1")

    # Seems like this just logs a lot of not necessary stuff. Might be needed later. But for now, we disable it.
    #if DEBUGGING:
    #    logging.basicConfig(level=logging.DEBUG)  # This enables detailed logs

    # Initialize model parameters
    net = get_model(MODEL_NAME)
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)


    # Define strategy
    strategy = CustomFedAvg(
        save_dir=os.path.join(DATA_ROOT, "torch_models", f"federated_{experiment}"),
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        on_fit_config_fn=on_fit_config,
        on_evaluate_config_fn=on_evaluate_config,
        lr_factor=0.5,        # Reduce LR by 50%
        lr_patience=5,        # Wait 5 rounds
        min_lr=0.000001       # Minimum LR
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
