"""Fine-tune an ImageNet-pretrained ResNet-18 for SIPaKMeD cell classification.

The raw images are never changed.  Loading, resizing, augmentation and ImageNet
normalization are performed in memory, complementing preprocessing/preprocess.py.

Run from the project root:
    python models/baseline_resnet.py --data-dir data/archive
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


CLASS_NAMES = [
    "Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal",
    "Superficial-Intermediate",
]
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def label_from_path(path: Path) -> str | None:
    """Map SIPaKMeD folders such as im_Koilocytotic to readable labels."""
    for part in path.parts:
        normalized = part.lower().replace("im_", "").replace("_", "-")
        for label in CLASS_NAMES:
            if normalized == label.lower():
                return label
    return None


def discover_images(data_dir: Path) -> pd.DataFrame:
    """Find valid cell images in the dataset's supplied CROPPED directories."""
    rows: list[dict[str, str]] = []
    for cropped_dir in sorted(path for path in data_dir.rglob("CROPPED") if path.is_dir()):
        for path in sorted(cropped_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = label_from_path(path)
            if label is None:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, ValueError):
                print(f"Skipping unreadable image: {path}")
                continue
            rows.append({"path": str(path.resolve()), "label": label})

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No readable images were found in CROPPED folders under {data_dir}")
    missing = [name for name in CLASS_NAMES if name not in set(frame["label"])]
    if missing:
        raise RuntimeError(f"Missing expected classes: {', '.join(missing)}")
    return frame


def make_splits(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified 70/15/15 train, validation and held-out test splits."""
    train_val, test = train_test_split(frame, test_size=0.15, stratify=frame["label"], random_state=seed)
    train, validation = train_test_split(
        train_val, test_size=0.15 / 0.85, stratify=train_val["label"], random_state=seed
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


class CellDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.label_to_index = {label: index for index, label in enumerate(CLASS_NAMES)}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            image = image.convert("RGB")
        return self.transform(image), self.label_to_index[row.label]


def make_loaders(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
                 image_size: int, batch_size: int, workers: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    evaluation_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    training_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    options = {"batch_size": batch_size, "num_workers": workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(CellDataset(train, training_transform), shuffle=True, **options),
        DataLoader(CellDataset(validation, evaluation_transform), shuffle=False, **options),
        DataLoader(CellDataset(test, evaluation_transform), shuffle=False, **options),
    )


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    
    predictions, targets = [], []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.tolist())
    return {"accuracy": accuracy_score(targets, predictions), "macro_f1": f1_score(targets, predictions, average="macro"), "predictions": predictions, "targets": targets}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained ResNet-18 baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/archive"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline_resnet"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224, help="ResNet's standard ImageNet input size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0, help="Use 0 on Windows unless worker processes are needed.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = discover_images(args.data_dir)
    train, validation, test = make_splits(frame, args.seed)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        split.to_csv(args.output_dir / f"{name}_manifest.csv", index=False)
    print(f"Using {len(frame)} cropped images: {dict(sorted(Counter(frame.label).items()))}")
    print(f"Split sizes - train: {len(train)}, validation: {len(validation)}, test: {len(test)}")
    print(f"Training on: {device}")

    train_loader, validation_loader, test_loader = make_loaders(train, validation, test, args.image_size, args.batch_size, args.workers)
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_state, best_f1, history = None, float("-inf"), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device, non_blocking=True)), labels.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        metrics = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_accuracy": metrics["accuracy"], "validation_macro_f1": metrics["macro_f1"]}
        history.append(row)
        print(json.dumps(row))
        if metrics["macro_f1"] > best_f1:
            best_f1, best_state = metrics["macro_f1"], copy.deepcopy(model.state_dict())

    torch.save({"model_state_dict": best_state, "class_names": CLASS_NAMES, "image_size": args.image_size, "architecture": "resnet18"}, args.output_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    model.load_state_dict(best_state)
    metrics = evaluate(model, test_loader, device)
    report = {
        "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
        "classification_report": classification_report(metrics["targets"], metrics["predictions"], target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(metrics["targets"], metrics["predictions"], labels=range(len(CLASS_NAMES))).tolist(),
    }
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"Saved best checkpoint and test metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
