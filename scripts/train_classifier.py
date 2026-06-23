from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from PIL import Image  # noqa: F401  # Windows DLL preload.
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from timm import create_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


MODEL_PRESETS = {
    "mambavision": "mamba_vision_T",
    "vmamba": "vmamba_tiny_s1l8",
    "hornet": "hornet_tiny_7x7",
    "fastvit": "fastvit_t8",
    "repvit": "repvit_m0_9",
    "maxvit": "maxvit_tiny_rw_224",
    "coatnet": "coatnet_0_rw_224",
    "edgenext": "edgenext_x_small",
    "focalnet": "focalnet_tiny_srf",
    "poolformer": "poolformer_s12",
    "tinyvit": "tiny_vit_5m_224",
    "siglip": "vit_base_patch16_siglip_224",
    "clip": "vit_base_patch16_clip_224",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_any_model(model_key: str, model_name: str, pretrained: bool, num_classes: int):
    if model_key == "mambavision":
        root = Path(__file__).resolve().parents[1]
        external_root = root / "external" / "MambaVision"
        if str(external_root) not in sys.path:
            sys.path.insert(0, str(external_root))
        from mambavision.models.mamba_vision import mamba_vision_T

        return mamba_vision_T(
            pretrained=pretrained,
            num_classes=num_classes,
            model_path=str(root / "experiments" / "classification" / "mambavision" / "pretrained_mambavision_tiny_1k.pth.tar"),
        )
    if model_key == "vmamba":
        root = Path(__file__).resolve().parents[1]
        external_root = root / "external" / "VMamba"
        if str(external_root) not in sys.path:
            sys.path.insert(0, str(external_root))
        import vmamba

        model = vmamba.vmamba_tiny_s1l8(pretrained=pretrained)
        model.classifier.head = nn.Linear(model.num_features, num_classes)
        return model
    if model_key == "hornet":
        root = Path(__file__).resolve().parents[1]
        external_root = root / "external" / "HorNet"
        if str(external_root) not in sys.path:
            sys.path.insert(0, str(external_root))
        import hornet

        return hornet.hornet_tiny_7x7(pretrained=False, num_classes=num_classes)
    return create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def build_loaders(data_dir: Path, image_size: int, batch_size: int, workers: int):
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tfms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, train: bool, use_amp: bool, scaler):
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits["logits"]
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for images, y in tqdm(loader, leave=False):
            images = images.to(device)
            logits = model(images)
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(y.tolist())
    elapsed = time.perf_counter() - start
    return preds, labels, elapsed / max(1, len(labels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True, choices=sorted(MODEL_PRESETS))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--data-dir", default="datasets/processed/classification_5class")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    model_name = args.model_name or MODEL_PRESETS[args.model_key]
    output_dir = Path(args.output_dir or f"experiments/classification/{args.model_key}")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(
        Path(args.data_dir), args.image_size, args.batch_size, args.workers
    )
    model = create_any_model(args.model_key, model_name, not args.no_pretrained, len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_acc = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True, use_amp, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False, use_amp, scaler)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        print(row)
        history.append(row)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": train_ds.classes, "model_name": model_name}, output_dir / "best.pt")

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    preds, labels, sec_per_image = evaluate(model, test_loader, device)
    report = classification_report(labels, preds, target_names=test_ds.classes, output_dict=True)
    cm = confusion_matrix(labels, preds).tolist()

    metrics = {
        "model_key": args.model_key,
        "model_name": model_name,
        "pretrained": not args.no_pretrained,
        "amp": use_amp,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "device": str(device),
        "classes": train_ds.classes,
        "best_val_acc": best_acc,
        "test_report": report,
        "confusion_matrix": cm,
        "seconds_per_image": sec_per_image,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["test_report"]["macro avg"], indent=2))


if __name__ == "__main__":
    main()
