from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from demo_web.app import create_app
from demo_web.inference.pipeline import InferencePipeline
from demo_web.inference.schemas import Classification, Detection


class FakeModels:
    device = "cpu"
    detectors = {
        "rfdetr_small": SimpleNamespace(label="RF-DETR Small", map50_95=0.7524),
        "yolov8s": SimpleNamespace(label="YOLOv8s", map50_95=0.6093),
    }
    classifiers = {
        "focalnet": SimpleNamespace(label="FocalNet", macro_f1=0.9937),
        "edgenext": SimpleNamespace(label="EdgeNeXt", macro_f1=0.9906),
    }

    def status(self) -> dict:
        return {
            "device": "cpu",
            "cuda_available": False,
            "detectors": {},
            "classifiers": {},
        }


class FakePipeline:
    models = FakeModels()

    def predict(
        self,
        image,
        filename,
        detector_model,
        classifier_model,
        mode,
        threshold,
    ):
        return {
            "request_id": "abc123",
            "filename": filename,
            "detector_model": detector_model,
            "detector_label": "YOLOv8s" if mode != "classification" else None,
            "classifier_model": classifier_model,
            "classifier_label": "EdgeNeXt" if mode != "detection" else None,
            "mode": mode,
            "threshold": threshold,
            "device": "cpu",
            "panel_count": 0,
            "message": "Panel bulunamadi.",
            "panels": [],
            "timings": {
                "detection_seconds": 0.01,
                "classification_seconds": 0.0,
                "total_seconds": 0.01,
            },
            "annotated_image_url": "/outputs/abc123/annotated.jpg",
            "json_url": "/outputs/abc123/result.json",
        }


class FailingPipeline(FakePipeline):
    def predict(self, *args, **kwargs):
        raise RuntimeError("C:\\Users\\example\\private-model.pt")


class FakeDetector:
    def predict(self, image, threshold):
        return [Detection((10, 10, 90, 70), 0.91)]


class FakeClassifier:
    def predict(self, images):
        return [
            Classification(
                class_name="dust",
                confidence=0.88,
                probabilities={
                    "bird_drop": 0.01,
                    "clean": 0.05,
                    "crack_or_damage": 0.02,
                    "dust": 0.88,
                    "snow": 0.04,
                },
            )
            for _ in images
        ]


class FakeModelManager:
    device = "cpu"
    detectors = {"yolov8s": SimpleNamespace(label="YOLOv8s")}
    classifiers = {"edgenext": SimpleNamespace(label="EdgeNeXt")}

    def get_detector(self, name):
        return FakeDetector()

    def get_classifier(self, name):
        return FakeClassifier()


def image_bytes(fmt: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (120, 80), "#234d3c").save(buffer, format=fmt)
    return buffer.getvalue()


def test_index_and_health() -> None:
    client = TestClient(create_app(FakePipeline()))
    response = client.get("/")
    assert response.status_code == 200
    assert "Güneş Paneli Analiz Sistemi" in response.text
    assert "YOLOv8s · %60.93 · Hızlı · ✓ Önerilen · Pratik" in response.text
    assert "RF-DETR Small · %75.24 · Orta · ✓ Önerilen · En güçlü" in response.text
    assert "EdgeNeXt · %99.06 · Hızlı · ✓ Önerilen · Pratik" in response.text
    assert "FocalNet · %99.37 · Orta · ✓ Önerilen · En güçlü" in response.text
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["models"]["device"] == "cpu"


def test_predict_validates_and_accepts_image() -> None:
    client = TestClient(create_app(FakePipeline()))
    response = client.post(
        "/api/predict",
        files={"file": ("panel.jpg", image_bytes(), "image/jpeg")},
        data={
            "detector_model": "yolov8s",
            "classifier_model": "edgenext",
            "mode": "full",
            "threshold": "0.25",
        },
    )
    assert response.status_code == 200
    assert response.json()["filename"] == "panel.jpg"

    invalid = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert invalid.status_code == 415


def test_predict_does_not_expose_internal_error_details() -> None:
    client = TestClient(create_app(FailingPipeline()))
    response = client.post(
        "/api/predict",
        files={"file": ("panel.jpg", image_bytes(), "image/jpeg")},
        data={
            "detector_model": "yolov8s",
            "classifier_model": "edgenext",
            "mode": "full",
            "threshold": "0.25",
        },
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Secilen model su anda kullanilamiyor."
    assert "private-model.pt" not in response.text


def test_pipeline_writes_annotated_result(tmp_path: Path) -> None:
    pipeline = InferencePipeline(tmp_path, tmp_path / "outputs")
    pipeline.models = FakeModelManager()
    result = pipeline.predict(
        image=Image.new("RGB", (120, 80), "#234d3c"),
        filename="panel.jpg",
        detector_model="yolov8s",
        classifier_model="edgenext",
        mode="full",
        threshold=0.25,
    )
    output_dir = tmp_path / "outputs" / result["request_id"]
    assert result["panel_count"] == 1
    assert result["panels"][0]["class_name"] == "dust"
    assert (output_dir / "annotated.jpg").exists()
    assert (output_dir / "panel_1.jpg").exists()
    assert (output_dir / "result.json").exists()


def test_pipeline_detection_only(tmp_path: Path) -> None:
    pipeline = InferencePipeline(tmp_path, tmp_path / "outputs")
    pipeline.models = FakeModelManager()
    result = pipeline.predict(
        image=Image.new("RGB", (120, 80), "#234d3c"),
        filename="panel.jpg",
        detector_model="yolov8s",
        classifier_model="edgenext",
        mode="detection",
        threshold=0.25,
    )
    assert result["panel_count"] == 1
    assert result["classifier_model"] is None
    assert result["panels"][0]["class_name"] is None
    assert result["panels"][0]["probabilities"] == {}
