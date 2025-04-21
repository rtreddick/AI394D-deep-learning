# Homework 4, Part 1b: Transformer Planner PRD

**Date:** April 21, 2025

## 1. Goal

Implement and train a Transformer-based model (`TransformerPlanner`) to predict future vehicle waypoints based on ground truth lane boundary data. The model must achieve the performance metrics specified in the homework README.

## 2. Background

This task builds upon Part 1a (MLP Planner). Instead of an MLP, we will use a Transformer architecture, specifically inspired by the Perceiver model, to predict waypoints. The model will take ground truth lane boundaries as input, simulating a perfect perception system.

## 3. Requirements

### 3.1. Model (`TransformerPlanner` in `homework/models.py`)

*   **Input:**
    *   `track_left`: Tensor of shape `(B, n_track, 2)` representing left lane boundary points (`n_track=10`).
    *   `track_right`: Tensor of shape `(B, n_track, 2)` representing right lane boundary points (`n_track=10`).
*   **Output:**
    *   `waypoints`: Tensor of shape `(B, n_waypoints, 2)` representing predicted vehicle positions for the next `n_waypoints` steps (`n_waypoints=3`).
*   **Architecture:** Perceiver-inspired Transformer.
    *   **Input Encoding:** Process the `track_left` and `track_right` inputs into a suitable feature representation (e.g., concatenate and pass through an MLP or Linear layer). This forms the "byte array".
    *   **Waypoint Queries:** Use `torch.nn.Embedding` to create learnable query embeddings for the `n_waypoints` target waypoints. This forms the "latent array".
    *   **Cross-Attention:** Employ a Transformer decoder mechanism (e.g., `torch.nn.TransformerDecoder` or `torch.nn.TransformerDecoderLayer`) where the waypoint embeddings act as queries, and the encoded lane boundary features serve as keys and values.
    *   **Output Head:** A final layer (e.g., Linear) to project the decoder output to the desired `(B, n_waypoints, 2)` shape.

### 3.2. Data

*   Utilize the `RoadDataset` from `datasets/road_dataset.py`.
*   Input data processing is handled by `datasets/road_transforms.py:EgoTrackProcessor`.
*   Data fields used: `track_left`, `track_right`, `waypoints`, `waypoints_mask`.

### 3.3. Training

*   Implement a training loop (similar to previous homeworks or the Colab notebook).
*   **Loss Function:** Choose a suitable loss for comparing predicted and ground truth waypoints (e.g., `torch.nn.MSELoss`, `torch.nn.L1Loss`). Consider using the `waypoints_mask`.
*   **Optimizer:** Select an appropriate optimizer (e.g., Adam, AdamW).
*   **Hyperparameter Tuning:** Experiment with learning rate, weight decay, embedding dimensions, number of attention heads, number of decoder layers, etc. The README suggests this model may require more tuning than the MLP.
*   **Logging & Saving:** Log training progress (loss, metrics) and save the best performing model checkpoint.

### 3.4. Performance Metrics

*   **Longitudinal Error:** < 0.2
*   **Lateral Error:** < 0.6
*   Evaluation should use the metrics provided in `homework/metrics.py`.

### 3.5. Evaluation (Optional)

*   Visualize driving performance using `homework/supertux_utils/evaluate.py` with the `--model transformer_planner` flag.

## 4. Plan / Todo List

1.  **[ ] Setup:**
    *   Ensure dataset (`drive_data/`) is downloaded and correctly placed.
    *   Verify Python environment and dependencies (`requirements.txt`).
    *   Review Part 1a code (`MLPPlanner`, training script) for reference.
2.  **[ ] Model Implementation (`homework/models.py`):**
    *   Define the `TransformerPlanner` class inheriting `torch.nn.Module`.
    *   Implement the input encoding layer(s).
    *   Initialize `nn.Embedding` for waypoint queries.
    *   Implement the Transformer decoder/cross-attention block(s).
    *   Implement the output prediction head.
    *   Define the `forward` method orchestrating the data flow.
3.  **[ ] Training Script:**
    *   Create or adapt a training script (e.g., `train_transformer.py`).
    *   Instantiate `TransformerPlanner`, dataset, dataloader.
    *   Set up the chosen optimizer and loss function.
    *   Implement the main training loop:
        *   Epoch iteration.
        *   Batch iteration.
        *   Data loading and moving to device.
        *   Forward pass.
        *   Loss calculation.
        *   Backward pass.
        *   Optimizer step.
        *   Zero gradients.
    *   Integrate validation loop and metric calculation.
    *   Add logging (e.g., using TensorBoard or print statements).
    *   Implement model checkpoint saving (using `utils.save_model`).
4.  **[ ] Hyperparameter Tuning:**
    *   Run initial training experiments with default hyperparameters.
    *   Systematically tune learning rate, optimizer parameters, model dimensions (embedding size, hidden dims, heads, layers).
    *   Monitor validation metrics to guide tuning.
5.  **[ ] Evaluation:**
    *   Periodically run the grader (`python3 -m grader homework -v`) on saved checkpoints.
    *   (Optional) Run visualization (`python3 -m homework.supertux_utils.evaluate --model transformer_planner --track lighthouse`) to observe driving behavior.
6.  **[ ] Refinement:**
    *   Analyze performance bottlenecks or failure modes.
    *   Iterate on model architecture or training strategy if performance targets are not met.
7.  **[ ] Code Cleanup & Documentation:**
    *   Add comments explaining key parts of the `TransformerPlanner` and training script.
    *   Ensure code follows style guidelines.
8.  **[ ] Bundling:**
    *   Prepare the final model checkpoint for submission.
    *   Run `bundle.py` and verify the bundle using the grader.

## 5. File Location

This document (`prd.md`) should be saved in the `deeplearning/homework_04/` directory.
