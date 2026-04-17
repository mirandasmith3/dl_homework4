from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(self, n_track: int = 10, n_waypoints: int = 3):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = n_track * 2 * 2  # left + right, each (n_track, 2)
        output_dim = n_waypoints * 2

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, track_left, track_right, **kwargs):
        b = track_left.shape[0]

        center = (track_left + track_right) / 2.0
        width = track_right - track_left

        x = torch.cat([center, width], dim=-1)
        x = x.view(b, -1)

        out = self.model(x)  # (B, n_waypoints*2)

        return out.view(b, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(self, n_track=10, n_waypoints=3, d_model=64):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Encode (x, y) → embedding
        self.input_proj = nn.Linear(2, d_model)

        # Query embeddings (one per waypoint)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Output projection
        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, track_left, track_right, **kwargs):
        b = track_left.shape[0]

        # Combine tracks → (B, 20, 2)
        x = torch.cat([track_left, track_right], dim=1)

        # Encode → (B, 20, d_model)
        memory = self.input_proj(x)

        # Queries → (B, n_waypoints, d_model)
        query = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)

        # Transformer decoding (cross attention)
        out = self.decoder(query, memory)

        # Predict coordinates
        out = self.output_proj(out)  # (B, n_waypoints, 2)

        return out


class CNNPlanner(nn.Module):
    def __init__(self, n_waypoints=3):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # (96x128 → 48x64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → 24x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # → 12x16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # → 6x8
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, n_waypoints * 2),
        )

    def forward(self, image, **kwargs):
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        b = x.shape[0]

        x = self.cnn(x)
        x = x.view(b, -1)

        x = self.fc(x)

        return x.view(b, self.n_waypoints, 2)
    

class LinearPlanner(nn.Module):
    def __init__(self, n_track=10, n_waypoints=3):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.fc = nn.Linear(n_track * 4, n_waypoints * 2)  
        # 4 = (left x,y + right x,y)

    def forward(self, track_left, track_right):
        x = torch.cat([track_left, track_right], dim=-1)  # (B, 10, 4)
        x = x.reshape(x.shape[0], -1)  # (B, 40)

        x = self.fc(x)  # (B, 6)
        return x.view(-1, self.n_waypoints, 2)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
    "linear_planner": LinearPlanner,
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
