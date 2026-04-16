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


def train():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_dataset = RoadDataset(split="train")
    val_dataset = RoadDataset(split="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model
    model = get_model(args.model).to(device)

    # Loss (IMPORTANT: regression task)
    criterion = torch.nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)

            if args.model == "cnn":
                images = batch["image"].to(device)
                preds = model(images)
            else:
                preds = model(track_left, track_right)

            loss = criterion(preds, waypoints)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Train Loss = {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        long_err, lat_err = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)

                if args.model == "cnn":
                    images = batch["image"].to(device)
                    preds = model(images)
                else:
                    preds = model(track_left, track_right)

                long_err += compute_longitudinal_error(preds, waypoints)
                lat_err += compute_lateral_error(preds, waypoints)

        long_err /= len(val_loader)
        lat_err /= len(val_loader)

        print(f"Val Longitudinal Error: {long_err:.4f}")
        print(f"Val Lateral Error: {lat_err:.4f}")

    # Save model
    save_model(model, f"{args.model}_planner.pt")


if __name__ == "__main__":
    train()