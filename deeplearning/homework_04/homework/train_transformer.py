"""
Usage:
    python3 -m homework.train_planner --your_args here

Used GitHub Copilot for assistance.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .models import save_model, load_model
from .datasets.road_dataset import load_data
from . import metrics


class WeightedL1Loss(nn.Module):
    """
    Custom L1 loss function that applies different weights to lateral and longitudinal errors.
    L1 loss is more robust to outliers than MSE loss.
    
    Args:
        lateral_weight: Weight multiplier for lateral errors (index 0 in this codebase)
        longitudinal_weight: Weight multiplier for longitudinal errors (index 1 in this codebase)
    """
    def __init__(self, lateral_weight=2.0, longitudinal_weight=1.0):
        super().__init__()
        self.lateral_weight = lateral_weight
        self.longitudinal_weight = longitudinal_weight
        
    def forward(self, pred, target):
        # Calculate absolute error (L1)
        abs_error = torch.abs(pred - target)
        
        # Apply weights according to the project's convention:
        # index 0 = lateral, index 1 = longitudinal
        weights = torch.ones_like(abs_error)
        weights[..., 0] = self.lateral_weight      # index 0 is lateral in this codebase
        weights[..., 1] = self.longitudinal_weight # index 1 is longitudinal in this codebase
        
        weighted_abs_error = abs_error * weights
        
        return weighted_abs_error.mean()


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 20,
    lr: float = 1e-4,  # Updated default learning rate
    batch_size: int = 64,
    n_track: int = 10,
    n_waypoints: int = 3,
    seed: int = 2024,
    d_model: int = 64,
    nhead: int = 4,
    num_decoder_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    activation: str = "relu",
    **kwargs,
):
    # Set up device (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create log directory with timestamp
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = tb.SummaryWriter(log_dir)

    # Create model using load_model
    model = load_model(
        model_name,
        with_weights=False,
        n_track=n_track,
        n_waypoints=n_waypoints,
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )
    model = model.to(device)
    model.train()

    # Setup data loaders
    data_path = Path(__file__).parent.parent / "drive_data"
    train_data = load_data(
        data_path / "train",
        transform_pipeline="state_only",
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    val_data = load_data(
        data_path / "val",
        transform_pipeline="state_only",
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Loss function, optimizer, and scheduler
    # Using a weighted L1 loss with stronger emphasis on lateral errors
    loss_func = WeightedL1Loss(lateral_weight=5.0, longitudinal_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # Updated optimizer to AdamW
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.3  # More aggressive LR reduction
    )

    # Create metrics
    train_metrics = metrics.PlannerMetric()
    val_metrics = metrics.PlannerMetric()

    global_step = 0
    best_val_loss = float('inf')  # Initialize best validation loss
    patience_counter = 0
    patience = 3  # Number of epochs to wait before early stopping
    
    # Training loop
    for epoch in range(num_epoch):
        # Reset metrics at beginning of epoch
        train_metrics.reset()
        val_metrics.reset()

        model.train()
        train_losses = []
        
        # Training phase
        for batch in train_data:
            # Extract track boundaries and waypoints
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred_waypoints = model(track_left=track_left, track_right=track_right)
            
            # Apply mask to consider only valid waypoints
            mask = waypoints_mask.unsqueeze(-1)
            masked_pred = pred_waypoints * mask
            masked_target = waypoints * mask
            
            # Compute loss
            loss = loss_func(masked_pred, masked_target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            global_step += 1
            
            # Update training metrics using the PlannerMetric.add() method
            train_metrics.add(
                preds=pred_waypoints.detach(),
                labels=waypoints,
                labels_mask=waypoints_mask
            )
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_data:
                # Extract track boundaries and waypoints
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)
                
                # Forward pass
                pred_waypoints = model(track_left=track_left, track_right=track_right)
                
                # Apply mask to consider only valid waypoints
                mask = waypoints_mask.unsqueeze(-1)
                masked_pred = pred_waypoints * mask
                masked_target = waypoints * mask
                
                # Compute loss
                loss = loss_func(masked_pred, masked_target)
                val_losses.append(loss.item())
                
                # Update validation metrics using PlannerMetric.add() method
                val_metrics.add(
                    preds=pred_waypoints,
                    labels=waypoints,
                    labels_mask=waypoints_mask
                )
        
        # Average losses
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        # Calculate average training metrics
        train_metrics_avg = train_metrics.compute()

        # Compute average metrics
        val_metrics_avg = val_metrics.compute()
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Logging
        logger.add_scalar("train/loss", avg_train_loss, global_step)
        logger.add_scalar("val/loss", avg_val_loss, global_step)
        
        # Log training metrics
        for metric_name, metric_value in train_metrics_avg.items():
            logger.add_scalar(f"train/{metric_name}", metric_value, global_step)
            
        # Log validation metrics
        for metric_name, metric_value in val_metrics_avg.items():
            logger.add_scalar(f"val/{metric_name}", metric_value, global_step)
        
        # Print progress
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == num_epoch - 1:
            print(
                f"Epoch {epoch + 1:2d}/{num_epoch:2d}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_loss={avg_val_loss:.4f}, "
                f"val_l1_error={val_metrics_avg.get('l1_error', 0):.4f}, "
                f"val_long_error={val_metrics_avg.get('longitudinal_error', 0):.4f}, "
                f"val_lat_error={val_metrics_avg.get('lateral_error', 0):.4f}"
            )
        
        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), log_dir / f"{model_name}_best.th")
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment patience counter
            
        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Print final metrics using the last validation metrics we already calculated
    print("\n======= FINAL MODEL METRICS =======")
    print(f"L1 error: {val_metrics_avg.get('l1_error', 0):.4f}")
    print(f"Longitudinal error: {val_metrics_avg.get('longitudinal_error', 0):.4f}")
    print(f"Lateral error: {val_metrics_avg.get('lateral_error', 0):.4f}")
    print("==================================\n")
    
    # Load the best model before saving for grading
    best_model_path = log_dir / f"{model_name}_best.th"
    if best_model_path.exists():
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        
        # Run validation again to get metrics for the best model
        model.eval()
        best_val_metrics = metrics.PlannerMetric()
        
        with torch.no_grad():
            for batch in val_data:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)
                
                pred_waypoints = model(track_left=track_left, track_right=track_right)
                
                best_val_metrics.add(
                    preds=pred_waypoints,
                    labels=waypoints,
                    labels_mask=waypoints_mask
                )
        
        best_metrics = best_val_metrics.compute()
        print("\n======= BEST MODEL METRICS =======")
        print(f"L1 error: {best_metrics.get('l1_error', 0):.4f}")
        print(f"Longitudinal error: {best_metrics.get('longitudinal_error', 0):.4f}")
        print(f"Lateral error: {best_metrics.get('lateral_error', 0):.4f}")
        print("==================================\n")
    
    # Save final model for grading
    save_model(model)
    
    # Also save a copy in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}_final.th")
    print(f"Training complete. Model saved to {log_dir / f'{model_name}_final.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Common arguments
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="transformer_planner") # Changed default
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)  # Updated default learning rate
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2024)
    
    # Data related arguments (usually fixed for this task)
    parser.add_argument("--n_track", type=int, default=10)
    parser.add_argument("--n_waypoints", type=int, default=3)

    # Transformer specific arguments
    parser.add_argument("--d_model", type=int, default=64, help="Dimension of the transformer model")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_decoder_layers", type=int, default=2, help="Number of decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=128, help="Dimension of the feedforward network")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for transformer")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function (relu or gelu")

    # Removed MLP specific arguments: hidden_dim, num_layers, dropout_rate
    
    args = parser.parse_args()
    train(**vars(args))
