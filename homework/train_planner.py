import argparse
import torch
from torch.utils.data import DataLoader

from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner
from homework.datasets.road_dataset import RoadDataset
from homework.metrics import compute_longitudinal_error, compute_lateral_error
from homework.utils import save_model  # if provided


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="mlp",
                        choices=["mlp", "transformer", "cnn"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    return parser.parse_args()


def get_model(model_name):
    if model_name == "mlp":
        return MLPPlanner()
    elif model_name == "transformer":
        return TransformerPlanner()
    elif model_name == "cnn":
        return CNNPlanner()


def train(
    model_name="mlp_planner",
    transform_pipeline="state_only",
    num_workers=4,
    lr=1e-3,
    batch_size=128,
    num_epoch=20,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = RoadDataset(split="train")
    val_dataset = RoadDataset(split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = get_model(model_name).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)

            if model_name == "cnn_planner":
                images = batch["image"].to(device)
                preds = model(images)
            else:
                preds = model(track_left, track_right)

            loss = criterion(preds, waypoints)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Train Loss = {total_loss / len(train_loader):.4f}")

    save_model(model)


if __name__ == "__main__":
    train()