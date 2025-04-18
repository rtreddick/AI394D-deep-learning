"""
Usage:
    python3 -m homework.train_planner --your_args here
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


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 30,
    lr: float = 1e-3,
    batch_size: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 3,
    n_track: int = 10,
    n_waypoints: int = 3,
    seed: int = 2024,
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
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        **kwargs
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
    loss_func = nn.MSELoss()  # Mean Squared Error for waypoint prediction
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    # Create metrics
    train_metrics = metrics.PlannerMetric()
    val_metrics = metrics.PlannerMetric()

    global_step = 0
    best_val_loss = float('inf')  # Initialize best validation loss
    
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
            masked_pred = pred_waypoints * waypoints_mask.unsqueeze(-1)
            masked_target = waypoints * waypoints_mask.unsqueeze(-1)
            
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
                masked_pred = pred_waypoints * waypoints_mask.unsqueeze(-1)
                masked_target = waypoints * waypoints_mask.unsqueeze(-1)
                
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
                f"train_l1_error={train_metrics_avg.get('l1_error', 0):.4f}, "
                f"train_long_error={train_metrics_avg.get('longitudinal_error', 0):.4f}, "
                f"train_lat_error={train_metrics_avg.get('lateral_error', 0):.4f}, "
                f"val_l1_error={val_metrics_avg.get('l1_error', 0):.4f}, "
                f"val_long_error={val_metrics_avg.get('longitudinal_error', 0):.4f}, "
                f"val_lat_error={val_metrics_avg.get('lateral_error', 0):.4f}"
            )
        
        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), log_dir / f"{model_name}_best.th")
    
    # Save final model for grading
    save_model(model)
    
    # Also save a copy in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}_final.th")
    print(f"Training complete. Model saved to {log_dir / f'{model_name}_final.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="mlp_planner")
    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--n_track", type=int, default=10)
    parser.add_argument("--n_waypoints", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2024)
    
    args = parser.parse_args()
    train(**vars(args))
