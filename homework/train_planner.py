import torch
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path

from homework.models import (
    MLPPlanner,
    TransformerPlanner,
    CNNPlanner,
    LinearPlanner,
    save_model,
)

from homework.datasets.road_dataset import RoadDataset
from homework.metrics import PlannerMetric


def get_model(model_name):
    if model_name == "mlp_planner":
        return MLPPlanner()
    elif model_name == "transformer_planner":
        return TransformerPlanner()
    elif model_name == "cnn_planner":
        return CNNPlanner()
    elif model_name == "linear_planner":
        return LinearPlanner()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def build_dataset(root, transform_pipeline):
    """
    Each episode folder is a separate dataset item.
    We wrap them with ConcatDataset.
    """
    episode_paths = sorted([p for p in Path(root).iterdir() if p.is_dir()])

    datasets = [
        RoadDataset(str(ep), transform_pipeline=transform_pipeline)
        for ep in episode_paths
    ]

    return ConcatDataset(datasets)


def train(
    model_name="transformer_planner",
    transform_pipeline="state_only",
    num_workers=2,
    lr=5e-4,       # lower than 1e-3
    batch_size=128,
    num_epoch=100, # more epochs with cosine decay
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== DATA =====
    data_root = Path("drive_data")

    train_dataset = build_dataset(data_root / "train", transform_pipeline)
    val_dataset = build_dataset(data_root / "val", transform_pipeline)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ===== MODEL =====
    model = get_model(model_name).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)

    # ===== TRAIN LOOP =====
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            if model_name == "cnn_planner":
                images = batch["image"].to(device)
                preds = model(images)
            else:
                preds = model(track_left, track_right)

            # masked MSE loss
            mask = waypoints_mask[..., None]
            loss = ((preds - waypoints) ** 2 * mask).sum() / mask.sum()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Train Loss = {total_loss / len(train_loader):.4f}")

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
        scheduler.step()


    # ===== SAVE =====
    save_model(model)


if __name__ == "__main__":
    train()