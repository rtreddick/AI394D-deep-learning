# Homework 4, Part 1b: Transformer Planner - Detailed Todo List

This list breaks down the steps required to implement and train the `TransformerPlanner` based on the PRD (`prd.md`).

## Phase 1: Setup & Preparation

-   [X] **Verify Environment:**
    -   [X] Ensure the `drive_data` directory exists in `deeplearning/homework_04/` and contains the dataset. If not, run the download command from `README.md`.
    -   [X] Confirm Python environment is set up (e.g., using `poetry install` based on `pyproject.toml` or `pip install -r requirements.txt`).
-   [X] **Review Existing Code:**
    -   [X] Familiarize yourself with the `MLPPlanner` implementation in `homework/models.py`.
    -   [X] Review the existing training script (`homework/train_planner.py`) or the Colab notebook structure for training loops, data loading, metrics, and saving.
    -   [X] Understand how `homework/datasets/road_dataset.py` and `homework/datasets/road_transforms.py:EgoTrackProcessor` provide the `track_left`, `track_right`, `waypoints`, and `waypoints_mask` data.
    -   [X] Review `homework/metrics.py:PlannerMetric` for calculating evaluation metrics.

## Phase 2: Model Implementation (`homework/models.py`)

-   [X] **Define `TransformerPlanner` Class:**
    -   [X] Create the `TransformerPlanner` class inheriting from `torch.nn.Module`.
    -   [X] Define `__init__` method accepting hyperparameters like `n_track`, `n_waypoints`, `d_model`, `nhead`, `num_decoder_layers`, `dim_feedforward`, `dropout`.
-   [X] **Implement Input Encoding:**
    -   [X] In `__init__`, define layer(s) to process input track boundaries into a sequence of features.
        -   *Suggestion:* Concatenate `track_left` and `track_right` (`B, n_track * 2, 2`). Process each point pair (or individual point) through a Linear layer or small MLP to project its 2D coordinates (or 4D if paired) into the `d_model` dimension. The goal is to get a tensor representing the track features as a sequence, e.g., shape `(B, N, d_model)` where `N` is the sequence length (e.g., `N = n_track * 2`). This will serve as the `memory` (key/value) for the decoder.
-   [ ] **Implement Positional Encoding:**
    -   [ ] Consider adding positional encoding to the encoded track features before passing them to the Transformer decoder. This helps the model understand the order/position of the track points. You can use learned embeddings or fixed sinusoidal encodings.
-   [ ] **Implement Waypoint Queries:**
    -   [ ] In `__init__`, initialize `self.query_embed = nn.Embedding(n_waypoints, d_model)`. This will be the learnable "latent array" from Perceiver, serving as the `tgt` (query) for the decoder.
-   [ ] **Implement Transformer Decoder / Cross-Attention:**
    -   [ ] In `__init__`, define the Transformer decoder layer(s). Use `nn.TransformerDecoderLayer` and potentially wrap with `nn.TransformerDecoder`.
        -   Set `d_model`, `nhead`, `dim_feedforward`, `dropout`, `activation`.
        -   Crucially, set `batch_first=True` if your tensors are `(Batch, Seq, Dim)`.
-   [ ] **Implement Output Head:**
    -   [ ] In `__init__`, define a final `nn.Linear` layer to project the decoder output features (dimension `d_model`) to the required waypoint coordinate dimension (2).
-   [ ] **Implement `forward` Method:**
    -   [ ] Accept `track_left` (`B, n_track, 2`) and `track_right` (`B, n_track, 2`).
    -   [ ] Process inputs through the encoding layer(s) to get the key/value tensor (`memory`: `B, N, d_model`).
    -   [ ] **(If using)** Add positional encoding to `memory`.
    -   [ ] Generate the query tensor (`tgt`) from `self.query_embed`. You'll need to expand it for the batch size: `self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)`. Shape: `(B, n_waypoints, d_model)`.
    -   [ ] Pass `tgt` and `memory` through the `nn.TransformerDecoder` or `nn.TransformerDecoderLayer`(s).
    -   [ ] Pass the decoder output through the final linear head.
    -   [ ] Ensure the output tensor has the correct shape `(B, n_waypoints, 2)`.
-   [ ] **Register in `MODEL_FACTORY`:**
    -   [ ] Ensure `transformer_planner` maps to your `TransformerPlanner` class in the `MODEL_FACTORY` dictionary.

## Phase 3: Training Script (`homework/train_transformer.py`)

-   [ ] **Create/Adapt Training Script:**
    -   [ ] Duplicate `homework/train_planner.py` to `homework/train_transformer.py` or create a new script.
    -   [ ] Update `argparse` to include Transformer-specific hyperparameters (`d_model`, `nhead`, `num_decoder_layers`, `dim_feedforward`, `dropout`) and set the default `model_name` to `"transformer_planner"`.
        -   *Suggestion for defaults:* `d_model=64`, `nhead=4`, `num_decoder_layers=2`, `dim_feedforward=128`, `dropout=0.1`.
    -   [ ] Modify the `load_model` call to pass the new Transformer hyperparameters.
-   [ ] **Instantiate Model & Components:**
    -   [ ] Ensure the script instantiates `TransformerPlanner` correctly using the arguments.
    -   [ ] Set up `RoadDataset` loaders (train/val) using `state_only` transform pipeline.
-   [ ] **Loss Function:**
    -   [ ] Choose a loss function (e.g., `nn.MSELoss`, `nn.L1Loss`, or the `WeightedL1Loss` from `train_planner.py`).
    -   [ ] **Important:** Apply the `waypoints_mask` correctly during loss calculation to ignore invalid waypoints. `loss = loss_func(pred_waypoints[waypoints_mask], waypoints[waypoints_mask])` or multiply predictions/targets by the mask before calculating loss if the loss function expects full tensors.
-   [ ] **Optimizer & Scheduler:**
    -   [ ] Choose an optimizer (e.g., `torch.optim.AdamW` is often good for Transformers).
    -   [ ] Consider using a learning rate scheduler (e.g., `ReduceLROnPlateau` or a warmup scheduler).
-   [ ] **Implement Training Loop:**
    -   [ ] Iterate through epochs and batches.
    -   [ ] Move data to the correct `device`.
    -   [ ] Perform forward pass: `pred_waypoints = model(track_left=..., track_right=...)`.
    -   [ ] Calculate loss (applying mask).
    -   [ ] Perform backward pass (`loss.backward()`).
    -   [ ] Optimizer step (`optimizer.step()`).
    -   [ ] Zero gradients (`optimizer.zero_grad()`).
-   [ ] **Implement Validation Loop:**
    -   [ ] Run periodically (e.g., end of each epoch).
    -   [ ] Use `torch.no_grad()`.
    -   [ ] Calculate validation loss (applying mask).
    -   [ ] Calculate metrics using `PlannerMetric.add()` and `PlannerMetric.compute()`.
-   [ ] **Logging:**
    -   [ ] Set up TensorBoard logging (`torch.utils.tensorboard.SummaryWriter`).
    -   [ ] Log training loss, validation loss, and validation metrics (L1, longitudinal, lateral errors) per epoch/step.
    -   [ ] Log learning rate if using a scheduler.
-   [ ] **Model Checkpointing:**
    -   [ ] Save the model state dict (`model.state_dict()`) periodically, especially when validation performance improves (e.g., lowest validation loss or best lateral/longitudinal error).
    -   [ ] Keep track of the best model checkpoint path.
-   [ ] **Final Model Saving:**
    -   [ ] After training, load the *best* saved checkpoint.
    -   [ ] Use the provided `save_model(model)` function to save the final `transformer_planner.th` file in the `homework/` directory for the grader.

## Phase 4: Training & Tuning

-   [ ] **Initial Training Run:**
    -   [ ] Run `python3 -m homework.train_transformer` with default hyperparameters.
    -   [ ] Monitor TensorBoard logs for loss curves and metrics. Check for NaNs or exploding gradients.
-   [ ] **Hyperparameter Tuning:**
    -   [ ] Experiment with:
        -   Learning rate (`lr`)
        -   Optimizer (AdamW vs Adam, weight decay)
        -   Model dimensions (`d_model`, `nhead`, `num_decoder_layers`, `dim_feedforward`)
        -   Dropout rates
        -   Batch size
        -   Loss function weighting (if using `WeightedL1Loss`)
    -   [ ] Keep track of experiments and their results. Aim to minimize validation loss and meet the target metrics.

## Phase 5: Evaluation & Refinement

-   [ ] **Grader Evaluation:**
    -   [ ] Run `python3 -m grader homework -v` on your saved `transformer_planner.th`.
    -   [ ] Check if performance targets are met:
        -   Longitudinal error < 0.2
        -   Lateral error < 0.6
-   [ ] **(Optional) Driving Visualization:**
    -   [ ] If PySuperTuxKart is installed, run `python3 -m homework.supertux_utils.evaluate --model transformer_planner --track lighthouse` (or other tracks) to observe driving behavior.
-   [ ] **Refinement:**
    -   [ ] If metrics are not met, analyze results. Is the model underfitting (high loss)? Overfitting (large gap between train/val loss)? Are errors higher laterally or longitudinally?
    -   [ ] Revisit model architecture (e.g., more layers, different embedding size) or training hyperparameters based on analysis. Iterate on training and tuning.

## Phase 6: Final Steps

-   [ ] **Code Cleanup:**
    -   [ ] Add comments and docstrings to `TransformerPlanner` and `train_transformer.py`.
    -   [ ] Ensure code is clean and readable. Remove unused imports or variables.
-   [ ] **Final Model Check:**
    -   [ ] Ensure the *best performing* model is saved as `homework/transformer_planner.th`.
-   [ ] **Bundling:**
    -   [ ] Run `python3 bundle.py homework $YOUR_UT_ID`.
    -   [ ] **Crucially:** Verify the bundle by running `python3 -m grader $YOUR_UT_ID.zip -v`. Ensure it passes the `transformer_planner` tests with the required metrics.
-   [ ] **Submission:**
    -   [ ] Submit the verified `.zip` file.

