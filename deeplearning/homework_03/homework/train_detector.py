# used AI coding assistant

import argparse
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric


def train(
    exp_dir: str = "logs",
    model_name: Literal["classifier", "detector"] = "detector",
    num_epoch: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    data_path = Path(__file__).parent.parent / "drive_data"
    train_data = load_data(data_path / "train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data(data_path / "val", shuffle=False)

    # create loss functions, optimizer, and metrics
    seg_loss_func = torch.nn.CrossEntropyLoss()
    depth_loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.3)
    
    # Create metrics
    train_metrics = DetectionMetric()
    val_metrics = DetectionMetric()

    global_step = 0
    best_val_iou = 0.0

    # training loop
    for epoch in range(num_epoch):
        # reset metrics at beginning of epoch
        train_metrics.reset()
        val_metrics.reset()

        model.train()
        train_seg_losses = []
        train_depth_losses = []

        for batch in train_data:
            # Move data to device
            image = batch["image"].to(device)
            track = batch["track"].to(device)
            depth = batch["depth"].to(device)

            # training step
            optimizer.zero_grad()
            seg_logits, pred_depth = model(image)
            
            # Compute losses
            seg_loss = seg_loss_func(seg_logits, track)
            depth_loss = depth_loss_func(pred_depth, depth)
            total_loss = seg_loss + depth_loss  # You may want to weight these differently
            
            total_loss.backward()
            optimizer.step()

            # Get predictions for metrics
            pred_labels = seg_logits.argmax(dim=1)
            train_metrics.add(pred_labels, track, pred_depth, depth)
            
            train_seg_losses.append(seg_loss.item())
            train_depth_losses.append(depth_loss.item())
            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
            val_seg_losses = []
            val_depth_losses = []

            for batch in val_data:
                image = batch["image"].to(device)
                track = batch["track"].to(device)
                depth = batch["depth"].to(device)

                seg_logits, pred_depth = model(image)
                
                # Compute validation losses
                seg_loss = seg_loss_func(seg_logits, track)
                depth_loss = depth_loss_func(pred_depth, depth)
                
                # Get predictions for metrics
                pred_labels = seg_logits.argmax(dim=1)
                val_metrics.add(pred_labels, track, pred_depth, depth)
                
                val_seg_losses.append(seg_loss.item())
                val_depth_losses.append(depth_loss.item())

        # Compute metrics
        train_results = train_metrics.compute()
        val_results = val_metrics.compute()

        # Update learning rate based on IoU
        val_iou = val_results["iou"]
        scheduler.step(val_iou)

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_model(model)
            torch.save(model.state_dict(), log_dir / f"{model_name}_best.th")

        # Log metrics to tensorboard
        logger.add_scalar("train/seg_loss", np.mean(train_seg_losses), global_step)
        logger.add_scalar("train/depth_loss", np.mean(train_depth_losses), global_step)
        logger.add_scalar("train/iou", train_results["iou"], global_step)
        logger.add_scalar("train/depth_error", train_results["abs_depth_error"], global_step)
        logger.add_scalar("train/tp_depth_error", train_results["tp_depth_error"], global_step)

        logger.add_scalar("val/seg_loss", np.mean(val_seg_losses), global_step)
        logger.add_scalar("val/depth_loss", np.mean(val_depth_losses), global_step)
        logger.add_scalar("val/iou", val_results["iou"], global_step)
        logger.add_scalar("val/depth_error", val_results["abs_depth_error"], global_step)
        logger.add_scalar("val/tp_depth_error", val_results["tp_depth_error"], global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_iou={train_results['iou']:.4f} "
                f"val_iou={val_results['iou']:.4f} "
                f"train_depth_err={train_results['abs_depth_error']:.4f} "
                f"val_depth_err={val_results['abs_depth_error']:.4f}"
            )

    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Model saved to {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["classifier", "detector"],
        default="detector"
    )
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))
