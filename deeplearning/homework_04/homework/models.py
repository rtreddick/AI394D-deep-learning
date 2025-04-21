from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F  # Added import

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout_rate: float = 0.3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            hidden_dim (int): size of hidden layers
            num_layers (int): number of hidden layers
            dropout_rate (float): dropout probability for regularization
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        
        # Input dimension: Each track has n_track points with 2 coordinates (x, y) for both left and right
        input_dim = n_track * 2 * 2
        output_dim = n_waypoints * 2
        
        # Create MLP layers with dropout and batch normalization for regularization
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))  # Add dropout after activation
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))  # Add dropout after each activation
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]
        
        # Flatten the track points
        # (b, n_track, 2) -> (b, n_track * 2)
        left_flat = track_left.reshape(batch_size, -1)
        right_flat = track_right.reshape(batch_size, -1)
        
        # Concatenate left and right track points
        # (b, n_track * 2) + (b, n_track * 2) -> (b, n_track * 4)
        track_flat = torch.cat([left_flat, right_flat], dim=1)
        
        # Process through MLP
        # (b, n_track * 4) -> (b, n_waypoints * 2)
        output = self.mlp(track_flat)
        
        # Reshape output to desired waypoints format
        # (b, n_waypoints * 2) -> (b, n_waypoints, 2)
        waypoints = output.reshape(batch_size, self.n_waypoints, 2)
        
        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,       # Dimension of the transformer model
        nhead: int = 4,          # Number of attention heads
        num_decoder_layers: int = 2, # Number of decoder layers
        dim_feedforward: int = 128, # Dimension of the feedforward network
        dropout: float = 0.1,    # Dropout rate
    ):
        """
        Transformer-based planner using cross-attention.

        Args:
            n_track (int): Number of points in each side of the track boundary input.
            n_waypoints (int): Number of future waypoints to predict.
            d_model (int): The dimensionality of the embeddings and transformer layers.
            nhead (int): The number of heads in the multiheadattention models.
            num_decoder_layers (int): The number of sub-decoder-layers in the decoder.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model
        self.input_seq_len = n_track * 2 # Combined length of left and right track points

        # 1. Input Encoding Layer
        self.input_proj = nn.Linear(2, d_model) # Project 2D coordinates to d_model

        # 2. Positional Encoding for Track Points (Learnable)
        # We have n_track points for left and n_track for right, concatenated
        self.pos_embed = nn.Embedding(self.input_seq_len, d_model)

        # 3. Waypoint Query Embeddings (Learnable)
        # These act as the initial state for the decoder (the 'latent array')
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # --- Transformer Decoder and Output Head will be added next ---

        # Placeholder for forward method
        # raise NotImplementedError

    # --- forward method will be defined later ---
    # def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
    #     raise NotImplementedError


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
