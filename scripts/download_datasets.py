from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


KAGGLE_DATASETS = [
    "pythonafroz/solar-panel-images",
    "alicjalena/pv-panel-defect-dataset",
]

ROBOFLOW_DATASETS = [
    # workspace, project, version, format
    ("solar-7u3z6", "solar-faults-detection", 2, "folder"),
    ("faultdetection-j9hnw", "solar-panel-pjsbe", 3, "folder"),
    ("solar-panel-4isfg", "custom-workflow-multi-label-classification-xy0sf", 1, "folder"),
]


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("Running:", " ".join(cmd[:3]), "...")
    subprocess.run(cmd, check=True, env=env)


def download_kaggle(dataset: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub

        path = Path(kagglehub.dataset_download(dataset))
        target = output_dir / dataset.replace("/", "__")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(path, target)
        print(f"Downloaded Kaggle dataset {dataset} -> {target}")
        return
    except Exception as exc:
        print(f"kagglehub failed for {dataset}: {exc}")

    try:
        run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset,
                "-p",
                str(output_dir / dataset.replace("/", "__")),
                "--unzip",
            ]
        )
        print(f"Downloaded Kaggle dataset {dataset}")
    except Exception as exc:
        print(f"Kaggle CLI failed for {dataset}: {exc}")


def download_roboflow(output_dir: Path) -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ROBOFLOW_API_KEY is not set; skipping Roboflow downloads.")
        return

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    for workspace, project, version, export_format in ROBOFLOW_DATASETS:
        target = output_dir / f"{workspace}__{project}__v{version}"
        try:
            rf.workspace(workspace).project(project).version(version).download(
                export_format,
                location=str(target),
                overwrite=True,
            )
            print(f"Downloaded Roboflow dataset {workspace}/{project}/{version} -> {target}")
        except Exception as exc:
            print(f"Roboflow download failed for {workspace}/{project}/{version}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="datasets/raw")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    kaggle_dir = raw_dir / "kaggle"
    roboflow_dir = raw_dir / "roboflow"

    for dataset in KAGGLE_DATASETS:
        download_kaggle(dataset, kaggle_dir)

    download_roboflow(roboflow_dir)


if __name__ == "__main__":
    main()
