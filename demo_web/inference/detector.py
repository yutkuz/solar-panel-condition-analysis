from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from .schemas import Detection


class BaseDetector:
    def predict(self, image: Image.Image, threshold: float) -> list[Detection]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class YoloDetector(BaseDetector):
    def __init__(self, checkpoint_path: Path, device: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO bagimliligi eksik. requirements-demo.txt dosyasini kurun."
            ) from exc

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"YOLO checkpoint bulunamadi: {checkpoint_path}")

        self.model = YOLO(str(checkpoint_path))
        self.device = device
        self._lock = threading.Lock()

    def predict(self, image: Image.Image, threshold: float) -> list[Detection]:
        with self._lock:
            result = self.model.predict(
                source=np.asarray(image),
                conf=threshold,
                iou=0.7,
                device=self.device,
                verbose=False,
            )[0]

        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        return [
            Detection(
                box=tuple(int(round(value)) for value in box),
                confidence=float(score),
            )
            for box, score in zip(boxes, scores)
        ]

    def close(self) -> None:
        try:
            self.model.to("cpu")
        except (AttributeError, RuntimeError):
            pass


class RFDetrDetector(BaseDetector):
    def __init__(self, checkpoint_path: Path, device: str) -> None:
        try:
            from rfdetr import RFDETRSmall
        except ImportError as exc:
            raise RuntimeError(
                "RF-DETR bagimliligi eksik. requirements-demo.txt dosyasini kurun."
            ) from exc

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"RF-DETR checkpoint bulunamadi: {checkpoint_path}")

        self.model = RFDETRSmall(
            pretrain_weights=str(checkpoint_path),
            num_classes=1,
            device=device,
        )
        if hasattr(self.model, "optimize_for_inference"):
            self.model.optimize_for_inference()
        self._lock = threading.Lock()

    def predict(self, image: Image.Image, threshold: float) -> list[Detection]:
        with self._lock:
            result = self.model.predict(image, threshold=threshold)

        boxes = np.asarray(result.xyxy, dtype=np.float32)
        scores = np.asarray(result.confidence, dtype=np.float32)
        return [
            Detection(
                box=tuple(int(round(value)) for value in box),
                confidence=float(score),
            )
            for box, score in zip(boxes, scores)
        ]

    def close(self) -> None:
        try:
            self.model.model.to("cpu")
        except (AttributeError, RuntimeError):
            pass


class EfficientDetDetector(BaseDetector):
    def __init__(
        self,
        checkpoint_path: Path,
        model_name: str,
        image_size: int,
        device: str,
    ) -> None:
        try:
            import torch
            from effdet import create_model
        except ImportError as exc:
            raise RuntimeError("EfficientDet bagimliligi eksik.") from exc

        self.device = device
        self.image_size = image_size
        self._torch = torch
        self._lock = threading.Lock()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = create_model(
            model_name,
            bench_task="predict",
            num_classes=1,
            pretrained=False,
        )
        self.model.load_state_dict(state["model"], strict=False)
        self.model.to(device)
        self.model.eval()

    def predict(self, image: Image.Image, threshold: float) -> list[Detection]:
        from torchvision.transforms import functional as transform

        resized = image.resize((self.image_size, self.image_size))
        tensor = transform.to_tensor(resized)
        tensor = transform.normalize(
            tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        with self._lock, self._torch.inference_mode():
            output = self.model(tensor.unsqueeze(0).to(self.device))[0]

        scale_x = image.width / self.image_size
        scale_y = image.height / self.image_size
        detections = []
        for row in output.detach().cpu().numpy():
            score = float(row[4])
            if score < threshold:
                continue
            detections.append(
                Detection(
                    box=(
                        int(round(row[0] * scale_x)),
                        int(round(row[1] * scale_y)),
                        int(round(row[2] * scale_x)),
                        int(round(row[3] * scale_y)),
                    ),
                    confidence=score,
                )
            )
        return detections

    def close(self) -> None:
        self.model.to("cpu")


class DeformableDetrDetector(BaseDetector):
    def __init__(self, model_path: Path, device: str) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
        except ImportError as exc:
            raise RuntimeError("Deformable DETR bagimliligi eksik.") from exc

        self.device = device
        self._torch = torch
        self._lock = threading.Lock()
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = DeformableDetrForObjectDetection.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    def predict(self, image: Image.Image, threshold: float) -> list[Detection]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._lock, self._torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = self._torch.tensor(
            [[image.height, image.width]],
            device=self.device,
        )
        result = self.processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )[0]
        return [
            Detection(
                box=tuple(int(round(value)) for value in box.tolist()),
                confidence=float(score),
            )
            for box, score in zip(result["boxes"].cpu(), result["scores"].cpu())
        ]

    def close(self) -> None:
        self.model.to("cpu")


class GroundingDinoDetector(BaseDetector):
    def __init__(self, model_path: Path, device: str) -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Grounding DINO bagimliligi eksik.") from exc

        self.device = device
        self._torch = torch
        self._lock = threading.Lock()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    def predict(self, image: Image.Image, threshold: float) -> list[Detection]:
        inputs = self.processor(
            images=image,
            text="solar panel.",
            return_tensors="pt",
        ).to(self.device)
        with self._lock, self._torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=0.05,
            target_sizes=[(image.height, image.width)],
        )[0]
        return [
            Detection(
                box=tuple(int(round(value)) for value in box.tolist()),
                confidence=float(score),
            )
            for box, score in zip(result["boxes"].cpu(), result["scores"].cpu())
        ]

    def close(self) -> None:
        self.model.to("cpu")


class YoloNasDetector(BaseDetector):
    def __init__(
        self,
        checkpoint_path: Path,
        project_root: Path,
        python_path: Path,
    ) -> None:
        worker_path = project_root / "demo_web" / "inference" / "yolonas_worker.py"
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["CRASH_HANDLER"] = "FALSE"
        self._lock = threading.Lock()
        self.process = subprocess.Popen(
            [
                str(python_path),
                str(worker_path),
                "--checkpoint",
                str(checkpoint_path),
            ],
            cwd=project_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        ready = self._read_response()
        if ready.get("status") != "ready":
            self.close()
            raise RuntimeError(ready.get("error", "YOLO-NAS worker baslatilamadi."))

    def _read_response(self) -> dict:
        if self.process.stdout is None:
            raise RuntimeError("YOLO-NAS worker ciktisi kullanilamiyor.")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("YOLO-NAS worker beklenmedik sekilde kapandi.")
        return json.loads(line)

    def predict(self, image: Image.Image, threshold: float) -> list[Detection]:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            temp_path = Path(temp.name)
        try:
            image.save(temp_path, format="JPEG", quality=95)
            with self._lock:
                if self.process.stdin is None:
                    raise RuntimeError("YOLO-NAS worker girdisi kullanilamiyor.")
                self.process.stdin.write(
                    json.dumps(
                        {"image_path": str(temp_path), "threshold": threshold}
                    )
                    + "\n"
                )
                self.process.stdin.flush()
                response = self._read_response()
            if "error" in response:
                raise RuntimeError(response["error"])
            return [
                Detection(
                    box=tuple(item["box"]),
                    confidence=float(item["confidence"]),
                )
                for item in response["detections"]
            ]
        finally:
            temp_path.unlink(missing_ok=True)

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write('{"command":"close"}\n')
                process.stdin.flush()
            process.wait(timeout=3)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
