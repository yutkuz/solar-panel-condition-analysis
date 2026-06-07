from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_web.inference.pipeline import InferencePipeline


def load_entries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["images"] if isinstance(data, dict) else data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harici demo setini uctan uca veya siniflandirma modunda degerlendirir."
    )
    parser.add_argument("--manifest", default="demo_samples/external/manifest.json")
    parser.add_argument("--detector-model", default="yolov8s")
    parser.add_argument("--classifier-model", default="edgenext")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument(
        "--output",
        default="reports/external_demo_evaluation.json",
    )
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    manifest_path = (project_root / args.manifest).resolve()
    entries = load_entries(manifest_path)
    pipeline = InferencePipeline(project_root, project_root / "demo_web/outputs")
    rows = []

    for entry in entries:
        local_path = project_root / entry["local_path"]
        mode = entry.get("mode", "full")
        image = ImageOps.exif_transpose(Image.open(local_path)).convert("RGB")
        started = time.perf_counter()
        result = pipeline.predict(
            image=image,
            filename=local_path.name,
            detector_model=args.detector_model,
            classifier_model=args.classifier_model,
            mode=mode,
            threshold=args.threshold,
        )
        predicted_classes = [panel["class_name"] for panel in result["panels"]]
        expected = entry["expected_class"]
        detected = result["panel_count"] > 0 if mode == "full" else None
        correct = expected in predicted_classes
        rows.append(
            {
                "id": entry["id"],
                "expected_class": expected,
                "mode": mode,
                "detected": detected,
                "panel_count": result["panel_count"],
                "predicted_classes": predicted_classes,
                "correct": correct,
                "total_seconds": round(time.perf_counter() - started, 4),
                "request_id": result["request_id"],
            }
        )
        print(f"{entry['id']}: {predicted_classes or ['panel_yok']} ({'dogru' if correct else 'yanlis'})")

    total = len(rows)
    full_flow_rows = [row for row in rows if row["mode"] == "full"]
    classification_rows = [row for row in rows if row["mode"] == "classification"]
    detected_count = sum(bool(row["detected"]) for row in full_flow_rows)
    correct_count = sum(row["correct"] for row in rows)
    per_class = defaultdict(
        lambda: Counter(total=0, correct=0, detection_applicable=0, detected=0)
    )
    for row in rows:
        stats = per_class[row["expected_class"]]
        stats["total"] += 1
        stats["correct"] += int(row["correct"])
        if row["detected"] is not None:
            stats["detection_applicable"] += 1
            stats["detected"] += int(row["detected"])

    report = {
        "detector_model": args.detector_model,
        "classifier_model": args.classifier_model,
        "threshold": args.threshold,
        "image_count": total,
        "cold_start_included": True,
        "full_flow_count": len(full_flow_rows),
        "full_flow_detection_rate": (
            detected_count / len(full_flow_rows) if full_flow_rows else 0.0
        ),
        "classification_only_count": len(classification_rows),
        "classification_only_accuracy": (
            sum(row["correct"] for row in classification_rows)
            / len(classification_rows)
            if classification_rows
            else 0.0
        ),
        "end_to_end_accuracy": correct_count / total if total else 0.0,
        "average_seconds": (
            sum(row["total_seconds"] for row in rows) / total if total else 0.0
        ),
        "per_class": {name: dict(stats) for name, stats in per_class.items()},
        "items": rows,
    }
    output_path = (project_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rapor: {output_path}")


if __name__ == "__main__":
    main()
