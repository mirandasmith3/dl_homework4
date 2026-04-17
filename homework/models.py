from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent

INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


# =========================
# MLP PLANNER
# =========================
class MLPPlanner(nn.Module):
    def __init__(self, n_track: int = 10, n_waypoints: int = 3):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = n_track * 2 * 2
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

        out = self.model(x)
        return out.view(b, self.n_waypoints, 2)


# =========================
# TRANSFORMER PLANNER
# =========================
class TransformerPlanner(nn.Module):
    def __init__(self, n_track=10, n_waypoints=3, d_model=64):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Compute lane center and width explicitly
        self.input_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Learnable positional encoding per track point
        self.pos_embed = nn.Embedding(n_track, d_model)

        self.query_embed = nn.Embedding(n_waypoints, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            batch_first=True,
            norm_first=True,  # pre-norm is more stable
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, track_left, track_right, **kwargs):
        b = track_left.shape[0]

        # Explicitly give model center and width as features
        center = (track_left + track_right) / 2.0  # (B, n_track, 2)
        width = (track_right - track_left)          # (B, n_track, 2)
        x = torch.cat([center, width], dim=-1)      # (B, n_track, 4)

        pos = self.pos_embed.weight.unsqueeze(0).expand(b, -1, -1)
        memory = self.input_proj(x) + pos

        query = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)

        out = self.decoder(query, memory)
        return self.output_proj(out)
    

# =========================
# CNN PLANNER
# =========================
class CNNPlanner(nn.Module):
    def __init__(self, n_waypoints=3):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.tensor(INPUT_STD))

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
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


# =========================
# LINEAR PLANNER
# =========================
class LinearPlanner(nn.Module):
    def __init__(self, n_track=10, n_waypoints=3):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.fc = nn.Linear(n_track * 4, n_waypoints * 2)

    def forward(self, track_left, track_right):
        x = torch.cat([track_left, track_right], dim=-1)
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        return x.view(x.shape[0], self.n_waypoints, 2)


# =========================
# FACTORY
# =========================
MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
    "linear_planner": LinearPlanner,
}


def load_model(model_name: str, with_weights: bool = False, **kwargs):
    m = MODEL_FACTORY[model_name](**kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        m.load_state_dict(torch.load(model_path, map_location="cpu"))

    return m


def save_model(model: nn.Module):
    for name, cls in MODEL_FACTORY.items():
        if isinstance(model, cls):
            path = HOMEWORK_DIR / f"{name}.th"
            torch.save(model.state_dict(), path)
            return path

    raise ValueError("Unknown model type")


def calculate_model_size_mb(model: nn.Module):
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024