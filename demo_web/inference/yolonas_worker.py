from __future__ import annotations

import argparse
import json
import sys
import traceback


def send(payload: dict, protocol) -> None:
    protocol.write(json.dumps(payload, ensure_ascii=True) + "\n")
    protocol.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    protocol = sys.stdout
    sys.stdout = sys.stderr

    try:
        from super_gradients.training import models

        model = models.get(
            "yolo_nas_s",
            num_classes=1,
            checkpoint_path=args.checkpoint,
        )
        send({"status": "ready"}, protocol)
    except Exception as exc:
        send({"status": "error", "error": str(exc)}, protocol)
        return 1

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                return 0
            prediction = model.predict(
                request["image_path"],
                conf=float(request["threshold"]),
            ).prediction
            detections = []
            for box, score in zip(
                prediction.bboxes_xyxy,
                prediction.confidence,
            ):
                detections.append(
                    {
                        "box": [int(round(float(value))) for value in box],
                        "confidence": float(score),
                    }
                )
            send({"detections": detections}, protocol)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            send({"error": str(exc)}, protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
