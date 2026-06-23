from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


TRIALS = [
    {"trial": 1, "lr": 1e-4, "batch_size": 1, "grad_accum_steps": 16},
    {"trial": 2, "lr": 5e-5, "batch_size": 1, "grad_accum_steps": 16},
    {"trial": 3, "lr": 1e-4, "batch_size": 2, "grad_accum_steps": 8},
]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def train_run(dataset_dir: Path, output_dir: Path, epochs: int, params: dict, accelerator: str) -> dict:
    from rfdetr import RFDETRSmall

    output_dir.mkdir(parents=True, exist_ok=True)
    model = RFDETRSmall()
    start = time.time()
    model.train(
        dataset_dir=str(dataset_dir),
        dataset_file="yolo",
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=params["batch_size"],
        grad_accum_steps=params["grad_accum_steps"],
        lr=params["lr"],
        num_workers=0,
        accelerator=accelerator,
        devices=1,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        class_names=["solar_panel"],
        run_test=True,
        progress_bar=None,
        tensorboard=False,
        wandb=False,
        checkpoint_interval=max(epochs, 1),
    )
    checkpoints = {
        name: str(output_dir / name)
        for name in [
            "checkpoint_best_total.pth",
            "checkpoint_best_ema.pth",
            "checkpoint_best_regular.pth",
            "checkpoint.pth",
        ]
        if (output_dir / name).exists()
    }
    return {
        **params,
        "accelerator": accelerator,
        "epochs": epochs,
        "seconds": round(time.time() - start, 2),
        "checkpoints": checkpoints,
    }


def train_with_fallback(dataset_dir: Path, output_dir: Path, epochs: int, params: dict, prefer_cuda: bool) -> dict:
    accelerators = ["gpu", "cpu"] if prefer_cuda else ["cpu"]
    last_error = None
    for accelerator in accelerators:
        try:
            return train_run(dataset_dir, output_dir / accelerator, epochs, params, accelerator)
        except Exception as exc:
            last_error = repr(exc)
            write_json(output_dir / f"{accelerator}_error.json", {"error": last_error, "params": params})
    raise RuntimeError(last_error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/processed/detection_solar_panel_standard_yolo")
    parser.add_argument("--output", default="experiments/detection/rfdetr_small")
    parser.add_argument("--tune-epochs", type=int, default=5)
    parser.add_argument("--final-epochs", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    prefer_cuda = args.device in {"auto", "cuda"}

    tuning_results = []
    for trial in TRIALS:
        write_json(status_path, {"model": "rfdetr_small", "stage": "tuning", "trial": trial["trial"]})
        trial_dir = output / "tuning" / f"trial_{trial['trial']:02d}"
        result = train_with_fallback(dataset_dir, trial_dir, args.tune_epochs, trial, prefer_cuda)
        tuning_results.append(result)

    # No comparable trial metric is saved here.
    best = tuning_results[0]
    write_json(output / "tuning_summary.json", {"trials": tuning_results, "best": best})
    write_json(status_path, {"model": "rfdetr_small", "stage": "final"})
    final = train_with_fallback(dataset_dir, output / "final", args.final_epochs, best, prefer_cuda)
    write_json(output / "final" / "metrics.json", {"model": "rfdetr_small", "status": "completed", **final})
    write_json(status_path, {"model": "rfdetr_small", "stage": "finished"})


if __name__ == "__main__":
    main()
