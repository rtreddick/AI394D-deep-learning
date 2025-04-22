"""
Usage:
    python3 -m homework.train_cnn --your_args here

Based on train_planner.py and train_transformer.py

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


def train(
    exp_dir: str = "logs",
    model_name: str = "cnn_planner", # Default to CNN planner
    num_epoch: int = 20,
    lr: float = 1e-4, # Adjusted learning rate for CNN
    batch_size: int = 32, # Smaller batch size might be needed for CNN memory
    seed: int = 2024,
    # CNNPlanner specific args (can be passed via kwargs)
    n_waypoints: int = 3,
    hidden_dim: int = 128,
    **kwargs,
):
    # Set up device (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA/MPS not available, using CPU")
        device = torch.device("cpu")

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create log directory with timestamp
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = tb.SummaryWriter(log_dir)

    # Create model using load_model
    # Pass CNN specific args if needed, rely on defaults otherwise
    model = load_model(
        model_name,
        with_weights=False,
        n_waypoints=n_waypoints,
        hidden_dim=hidden_dim,
        **kwargs # Pass any other potential model args
    )
    model = model.to(device)
    model.train()

    # Setup data loaders - Use 'default' pipeline to get images
    data_path = Path(__file__).parent.parent / "drive_data"
    train_data = load_data(
        data_path / "train",
        transform_pipeline="default", # Use default pipeline for images
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    val_data = load_data(
        data_path / "val",
        transform_pipeline="default", # Use default pipeline for images
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Loss function, optimizer, and scheduler
    # Use simple unweighted L1 loss for waypoint regression
    loss_func = nn.L1Loss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) # AdamW with small decay
    # Optional: Add scheduler like ReduceLROnPlateau if needed
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )

    # Create metrics
    train_metrics = metrics.PlannerMetric()
    val_metrics = metrics.PlannerMetric()

    global_step = 0
    best_val_loss = float('inf')  # Initialize best validation loss
    patience_counter = 0
    patience = 5 # Number of epochs to wait before early stopping

    # Training loop
    for epoch in range(num_epoch):
        # Reset metrics at beginning of epoch
        train_metrics.reset()
        val_metrics.reset()

        model.train()
        train_losses = []

        # Training phase
        for batch in train_data:
            # Extract image and waypoints
            image = batch["image"].to(device) # Use image input
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass using image
            pred_waypoints = model(image=image)

            # Apply mask to consider only valid waypoints
            masked_pred = pred_waypoints * waypoints_mask.unsqueeze(-1)
            masked_target = waypoints * waypoints_mask.unsqueeze(-1)

            # Compute loss (average over valid coordinates)
            err = loss_func(masked_pred, masked_target)          # (B, n_waypoints, 2)
            mask_f = waypoints_mask.unsqueeze(-1).float()       # (B, n_waypoints, 1)
            masked_err = err * mask_f                           # zero out invalid
            # Ensure mask_f.sum() is not zero to avoid division by zero
            valid_coords_count = mask_f.sum() * 2
            if valid_coords_count > 0:
                loss = masked_err.sum() / valid_coords_count
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True) # Handle cases with no valid waypoints

            # Backward pass and optimization
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            global_step += 1

            # Update training metrics
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
                # Extract image and waypoints
                image = batch["image"].to(device) # Use image input
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                # Forward pass using image
                pred_waypoints = model(image=image)

                # Apply mask to consider only valid waypoints
                masked_pred = pred_waypoints * waypoints_mask.unsqueeze(-1)
                masked_target = waypoints * waypoints_mask.unsqueeze(-1)

                # Compute loss (average over valid coordinates)
                err = loss_func(masked_pred, masked_target)       # (B, n_waypoints, 2)
                mask_f = waypoints_mask.unsqueeze(-1).float()    # (B, n_waypoints, 1)
                masked_err = err * mask_f                        # zero out invalid
                valid_coords_count = mask_f.sum() * 2
                if valid_coords_count > 0:
                    loss = masked_err.sum() / valid_coords_count
                else:
                    loss = torch.tensor(0.0, device=device) # Handle cases with no valid waypoints

                val_losses.append(loss.item())

                # Update validation metrics
                val_metrics.add(
                    preds=pred_waypoints,
                    labels=waypoints,
                    labels_mask=waypoints_mask
                )

        # Average losses
        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0

        # Calculate average metrics
        train_metrics_avg = train_metrics.compute()
        val_metrics_avg = val_metrics.compute()

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Logging
        logger.add_scalar("train/loss", avg_train_loss, global_step)
        logger.add_scalar("val/loss", avg_val_loss, global_step)
        logger.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)

        # Log training metrics
        for metric_name, metric_value in train_metrics_avg.items():
            logger.add_scalar(f"train/{metric_name}", metric_value, global_step)

        # Log validation metrics
        for metric_name, metric_value in val_metrics_avg.items():
            logger.add_scalar(f"val/{metric_name}", metric_value, global_step)

        # Print progress
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
            print(f"  -> Saved best model (val_loss: {best_val_loss:.4f})")
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment patience counter

        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in val_loss.")
            break

    # --- Post-Training ---

    # Load the best model before final evaluation and saving for grading
    best_model_path = log_dir / f"{model_name}_best.th"
    if best_model_path.exists():
        print(f"\nLoading best model from {best_model_path} for final evaluation.")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Run validation again to get metrics for the best model
        model.eval()
        best_val_metrics = metrics.PlannerMetric()
        final_val_losses = []

        with torch.no_grad():
            for batch in val_data:
                image = batch["image"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                pred_waypoints = model(image=image)

                # Loss calculation (same as validation loop)
                masked_pred = pred_waypoints * waypoints_mask.unsqueeze(-1)
                masked_target = waypoints * waypoints_mask.unsqueeze(-1)
                err = loss_func(masked_pred, masked_target)
                mask_f = waypoints_mask.unsqueeze(-1).float()
                masked_err = err * mask_f
                valid_coords_count = mask_f.sum() * 2
                if valid_coords_count > 0:
                    loss = masked_err.sum() / valid_coords_count
                else:
                    loss = torch.tensor(0.0, device=device)
                final_val_losses.append(loss.item())

                best_val_metrics.add(
                    preds=pred_waypoints,
                    labels=waypoints,
                    labels_mask=waypoints_mask
                )

        best_metrics = best_val_metrics.compute()
        final_avg_val_loss = sum(final_val_losses) / len(final_val_losses) if final_val_losses else 0
        print("\n======= BEST MODEL METRICS (Validation Set) =======")
        print(f"Validation Loss: {final_avg_val_loss:.4f}")
        print(f"L1 error: {best_metrics.get('l1_error', 0):.4f}")
        print(f"Longitudinal error: {best_metrics.get('longitudinal_error', 0):.4f}")
        print(f"Lateral error: {best_metrics.get('lateral_error', 0):.4f}")
        print("===================================================\n")
    else:
        print("\nNo best model saved. Using the model from the final epoch.")
        # If no best model was saved (e.g., training stopped early),
        # the current model state is from the last completed epoch.
        # We can report the metrics computed at the end of the last epoch.
        print("\n======= FINAL EPOCH METRICS (Validation Set) =======")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"L1 error: {val_metrics_avg.get('l1_error', 0):.4f}")
        print(f"Longitudinal error: {val_metrics_avg.get('longitudinal_error', 0):.4f}")
        print(f"Lateral error: {val_metrics_avg.get('lateral_error', 0):.4f}")
        print("====================================================\n")


    # Save final model for grading (using the best loaded weights if available)
    final_save_path = save_model(model)
    print(f"Final model saved for grading: {final_save_path}")

    # Also save a copy in the log directory (redundant if best model was loaded and saved)
    # torch.save(model.state_dict(), log_dir / f"{model_name}_final.th")
    print(f"Training complete. Logs and models saved in: {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN Planner Model")

    # Common arguments
    parser.add_argument("--exp_dir", type=str, default="logs", help="Directory for logs and saved models.")
    parser.add_argument("--model_name", type=str, default="cnn_planner", help="Name of the model to train.")
    parser.add_argument("--num_epoch", type=int, default=30, help="Number of training epochs.") # Increased epochs
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducibility.")

    # CNNPlanner specific arguments (matching defaults in CNNPlanner class)
    parser.add_argument("--n_waypoints", type=int, default=3, help="Number of waypoints to predict.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size in the FC layers.")

    # Add any other arguments needed, potentially passed via **kwargs to train()

    args = parser.parse_args()
    train(**vars(args))
