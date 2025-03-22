import argparse
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import ClassificationLoss, load_model, save_model
from .datasets.classification_dataset import load_data
from .metrics import AccuracyMetric


def train(
    exp_dir: str = "logs",
    model_name: Literal["classifier", "detector"] = "classifier",
    num_epoch: int = 20,
    lr: float = 1e-3,
    batch_size: int = 128,
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

    data_path = Path(__file__).parent.parent / "classification_data"
    train_data = load_data(data_path / "train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data(data_path / "val", shuffle=False)

    # create loss function, optimizer, and metrics
    loss_func = ClassificationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.3)
    
    # Create metrics
    train_accuracy = AccuracyMetric()
    val_accuracy = AccuracyMetric()

    global_step = 0

    # training loop
    for epoch in range(num_epoch):
        # reset metrics at beginning of epoch
        train_accuracy.reset()
        val_accuracy.reset()

        model.train()
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # training step
            optimizer.zero_grad()
            logits = model(img)
            loss = loss_func(logits, label)
            loss.backward()
            optimizer.step()

            train_accuracy.add(logits.argmax(dim=1), label)
            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                logits = model(img)
                val_accuracy.add(logits.argmax(dim=1), label)

        # Get accuracy values
        epoch_train_acc = train_accuracy.compute()["accuracy"]
        epoch_val_acc = val_accuracy.compute()["accuracy"]

        # Reduce LR if val_acc plateaus
        scheduler.step(epoch_val_acc)

        # log to tensorboard
        logger.add_scalar("train_acc", epoch_train_acc, global_step)
        logger.add_scalar("val_acc", epoch_val_acc, global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["classifier", "detector"],
        default="classifier"
    )
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))
