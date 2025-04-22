# Used GitHub Copilot for assistance.

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
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 1, # Start with one layer as per notes
        dim_feedforward: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Input embedding layer: projects 2D coordinates to d_model
        self.input_embed = nn.Linear(2, d_model)

        # Learnable query embeddings for the waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Transformer Decoder Layer setup
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Ensure batch dimension comes first
        )
        # Stacking decoder layers (using nn.TransformerDecoder)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection layer: maps d_model back to 2D coordinates
        self.output_proj = nn.Linear(d_model, 2)


    def forward(
        self,
        track_left: torch.Tensor, # Shape: [B, n_track, 2]
        track_right: torch.Tensor, # Shape: [B, n_track, 2]
        **kwargs,
    ) -> torch.Tensor:
        batch_size = track_left.shape[0]

        # 1. Embed track boundaries
        # (B, n_track, 2) -> (B, n_track, d_model)
        embedded_left = self.input_embed(track_left)
        embedded_right = self.input_embed(track_right)

        # 2. Concatenate to form memory
        # (B, n_track, d_model) + (B, n_track, d_model) -> (B, 2 * n_track, d_model)
        # memory shape: [B, 20, d_model]
        memory = torch.cat([embedded_left, embedded_right], dim=1)

        # 3. Prepare query embeddings (tgt)
        # Get the learnable query weights: [n_waypoints, d_model]
        query_embeddings = self.query_embed.weight
        # Expand across batch dimension: [1, n_waypoints, d_model] -> [B, n_waypoints, d_model]
        # tgt shape: [B, 3, d_model]
        tgt = query_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)

        # 4. Pass through Transformer Decoder
        # Input shapes: tgt=[B, n_waypoints, d_model], memory=[B, 2*n_track, d_model]
        # Output shape: [B, n_waypoints, d_model]
        output = self.transformer_decoder(tgt, memory)

        # 5. Project to output coordinates
        # (B, n_waypoints, d_model) -> (B, n_waypoints, 2)
        waypoints = self.output_proj(output)

        return waypoints # Shape: [B, 3, 2]


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
        hidden_dim: int = 128, # Added hidden dimension for FC layers
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Define the CNN backbone
        self.conv_layers = nn.Sequential(
            # Input: (B, 3, 96, 128)
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), # (B, 16, 48, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 16, 24, 32)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (B, 32, 24, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (B, 32, 12, 16)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (B, 64, 12, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (B, 64, 6, 8)
        )

        # Calculate the flattened size after conv layers
        # 64 channels * 6 height * 8 width
        flattened_size = 64 * 6 * 8

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_waypoints * 2) # Output: (B, n_waypoints * 2)
        )


    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        batch_size = image.shape[0]
        x = image
        # Normalize the image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Pass through CNN backbone
        x = self.conv_layers(x) # (B, 64, 6, 8)

        # Flatten the output for FC layers
        x = x.view(batch_size, -1) # (B, 64 * 6 * 8)

        # Pass through FC layers
        x = self.fc_layers(x) # (B, n_waypoints * 2)

        # Reshape to the desired output format
        waypoints = x.view(batch_size, self.n_waypoints, 2) # (B, n_waypoints, 2)

        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner, # Added comma
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Model '{model_name}' not found in MODEL_FACTORY")

    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        if not model_path.exists(): # Check existence properly
             raise FileNotFoundError(f"{model_path.name} not found") # Use FileNotFoundError

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

    for n, m_cls in MODEL_FACTORY.items(): # Corrected variable name
        if isinstance(model, m_cls): # Use isinstance for type checking
            model_name = n
            break # Found the model name, exit loop

    if model_name is None:
        raise ValueError(f"Model type '{type(model).__name__}' not supported") # Improved error message

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return str(output_path) # Return string path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
