import torch
from torch.utils.data import DataLoader

from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner
from homework.datasets.road_dataset import RoadDataset
from homework.metrics import PlannerMetric
from homework.models import save_model


def get_model(model_name):
    if model_name == "mlp_planner":
        return MLPPlanner()
    elif model_name == "transformer_planner":
        return TransformerPlanner()
    elif model_name == "cnn_planner":
        return CNNPlanner()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train(
    model_name="mlp_planner",
    transform_pipeline="state_only",  # not used but kept for compatibility
    num_workers=4,
    lr=1e-3,
    batch_size=128,
    num_epoch=20,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_dataset = RoadDataset(split="train")
    val_dataset = RoadDataset(split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Model
    model = get_model(model_name).to(device)

    # Loss + optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epoch):
        # ===== TRAIN =====
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

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")

        # ===== VALIDATION =====
        model.eval()
        metric = PlannerMetric()
        metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                if model_name == "cnn_planner":
                    images = batch["image"].to(device)
                    preds = model(images)
                else:
                    preds = model(track_left, track_right)

                metric.add(preds, waypoints, waypoints_mask)

        results = metric.compute()

        print(f"Val Longitudinal Error: {results['longitudinal_error']:.4f}")
        print(f"Val Lateral Error: {results['lateral_error']:.4f}")

    # Save model (grader expects this)
    save_model(model)


if __name__ == "__main__":
    train()