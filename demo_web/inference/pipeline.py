from __future__ import annotations

import gc
import importlib.util
import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from .classifier import PanelClassifier, PanelEnsembleClassifier
from .detector import (
    BaseDetector,
    DeformableDetrDetector,
    EfficientDetDetector,
    GroundingDinoDetector,
    RFDetrDetector,
    YoloDetector,
    YoloNasDetector,
)
from .schemas import CLASS_LABELS_TR, Detection, InferenceMode, PanelResult


@dataclass(frozen=True)
class ClassifierConfig:
    label: str
    checkpoint_path: Path
    macro_f1: float
    loader: str = "timm"
    image_size: int = 224
    family: str = "Guncel"
    members: tuple[str, ...] = ()
    weights: tuple[float, ...] = ()


@dataclass(frozen=True)
class DetectorConfig:
    label: str
    checkpoint_path: Path
    map50_95: float
    loader: str
    model_name: str | None = None
    image_size: int | None = None


class ModelManager:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        classification_root = project_root / "experiments" / "classification"
        detection_root = project_root / "experiments" / "detection"
        packaged_legacy_root = project_root / "legacy_models"
        existing_legacy_root = (
            project_root.parent
            / "gunes-paneli-projesi_ubuntudan_extracted"
            / "gunes-paneli-projesi"
        )
        legacy_root = (
            packaged_legacy_root
            if packaged_legacy_root.exists()
            else existing_legacy_root
        )
        legacy_comparison = legacy_root / "gunes-paneli-comparison" / "models"
        self.classifiers = {
            "focalnet": ClassifierConfig(
                "FocalNet", classification_root / "focalnet/final/best.pt", 0.9937
            ),
            "edgenext": ClassifierConfig(
                "EdgeNeXt", classification_root / "edgenext/final/best.pt", 0.9906
            ),
            "mambavision": ClassifierConfig(
                "MambaVision",
                classification_root / "mambavision/final/best.pt",
                0.9902,
                loader="mambavision",
            ),
            "fastvit": ClassifierConfig(
                "FastViT", classification_root / "fastvit/final/best.pt", 0.9831
            ),
            "repvit": ClassifierConfig(
                "RepViT", classification_root / "repvit/final/best.pt", 0.9786
            ),
            "tinyvit": ClassifierConfig(
                "TinyViT", classification_root / "tinyvit/final/best.pt", 0.9781
            ),
            "maxvit": ClassifierConfig(
                "MaxViT", classification_root / "maxvit/final/best.pt", 0.9750
            ),
            "poolformer": ClassifierConfig(
                "PoolFormer", classification_root / "poolformer/final/best.pt", 0.9697
            ),
            "coatnet": ClassifierConfig(
                "CoAtNet", classification_root / "coatnet/final/best.pt", 0.9564
            ),
            "siglip": ClassifierConfig(
                "SigLIP", classification_root / "siglip/final/best.pt", 0.8666
            ),
            "legacy_effnet_b3": ClassifierConfig(
                "EfficientNet-B3",
                legacy_root / "gunes-paneli-v2/models/best.pt",
                0.9846,
                loader="legacy_effnet_b3",
                family="Ubuntu",
            ),
            "legacy_convnext": ClassifierConfig(
                "ConvNeXt-Base",
                legacy_comparison / "convnext/best_convnext.pt",
                0.9738,
                loader="legacy_convnext",
                family="Ubuntu",
            ),
            "legacy_swinv2": ClassifierConfig(
                "SwinV2-B",
                legacy_comparison / "swinv2/best_swinv2.pt",
                0.9642,
                loader="legacy_swinv2",
                image_size=256,
                family="Ubuntu",
            ),
            "legacy_clip": ClassifierConfig(
                "CLIP ViT-L/14",
                legacy_comparison / "clip/best_clip.pt",
                0.9589,
                loader="legacy_clip",
                family="Ubuntu",
            ),
            "legacy_maxvit": ClassifierConfig(
                "MaxViT-Tiny",
                legacy_comparison / "maxvit/best_maxvit.pt",
                0.9551,
                loader="legacy_maxvit",
                family="Ubuntu",
            ),
            "legacy_dinov2b": ClassifierConfig(
                "DINOv2 ViT-B/14",
                legacy_comparison / "dinov2b/best_dinov2b.pt",
                0.9530,
                loader="legacy_dinov2b",
                family="Ubuntu",
            ),
            "legacy_effnetv2l": ClassifierConfig(
                "EfficientNetV2-L",
                legacy_comparison / "effnetv2l/best_effnetv2l.pt",
                0.9517,
                loader="legacy_effnetv2l",
                image_size=480,
                family="Ubuntu",
            ),
            "legacy_vitlarge": ClassifierConfig(
                "ViT-Large/16",
                legacy_comparison / "vitlarge/best_vitlarge.pt",
                0.9515,
                loader="legacy_vitlarge",
                family="Ubuntu",
            ),
            "legacy_dinov2s": ClassifierConfig(
                "DINOv2 ViT-S/14",
                legacy_comparison / "dinov2/best_dinov2.pt",
                0.9425,
                loader="legacy_dinov2s",
                family="Ubuntu",
            ),
        }
        self.classifiers["ensemble_effnet_swin"] = ClassifierConfig(
            "Ensemble: EfficientNet-B3 + SwinV2-B",
            legacy_root,
            0.9898,
            loader="ensemble",
            family="Ensemble",
            members=("legacy_effnet_b3", "legacy_swinv2"),
            weights=(0.571, 0.429),
        )
        self.detectors = {
            "rfdetr_small": DetectorConfig(
                "RF-DETR Small",
                detection_root
                / "rfdetr_small/final/gpu/checkpoint_best_total.pth",
                0.7524,
                "rfdetr",
            ),
            "deformable_detr": DetectorConfig(
                "Deformable DETR",
                detection_root / "deformable_detr/final",
                0.6232,
                "deformable_detr",
            ),
            "yolonas_s": DetectorConfig(
                "YOLO-NAS-S",
                detection_root
                / "yolonas_s/final/RUN_20260515_152826_439842/ckpt_best.pth",
                0.6119,
                "yolonas",
            ),
            "efficientdet_d2": DetectorConfig(
                "EfficientDet-D2",
                detection_root / "efficientdet_d2/final/best.pth",
                0.6094,
                "efficientdet",
                model_name="tf_efficientdet_d2",
                image_size=768,
            ),
            "yolov8s": DetectorConfig(
                "YOLOv8s",
                detection_root / "yolov8s/final/weights/best.pt",
                0.6093,
                "ultralytics",
            ),
            "rtdetr_l": DetectorConfig(
                "RT-DETR-L",
                detection_root / "rtdetr_l/final/weights/best.pt",
                0.6053,
                "ultralytics",
            ),
            "yolo11s": DetectorConfig(
                "YOLOv11s",
                detection_root / "yolo11s/final/weights/best.pt",
                0.6005,
                "ultralytics",
            ),
            "efficientdet_d1": DetectorConfig(
                "EfficientDet-D1",
                detection_root / "efficientdet_d1/final/best.pth",
                0.5950,
                "efficientdet",
                model_name="tf_efficientdet_d1",
                image_size=640,
            ),
            "yolo12s": DetectorConfig(
                "YOLOv12s",
                detection_root / "yolo12s/final/weights/best.pt",
                0.5813,
                "ultralytics",
            ),
            "grounding_dino_tiny": DetectorConfig(
                "Grounding DINO Tiny",
                detection_root / "grounding_dino_tiny/final",
                0.5806,
                "grounding_dino",
            ),
        }
        self._active_detector: tuple[str, BaseDetector] | None = None
        self._active_classifier: (
            tuple[str, PanelClassifier | PanelEnsembleClassifier] | None
        ) = None
        self._errors: dict[str, str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _yolonas_python_path() -> Path:
        configured = os.getenv("YOLO_NAS_PYTHON")
        if configured:
            return Path(configured).expanduser()
        return (
            Path.home()
            / "anaconda3"
            / "envs"
            / "solar-panel-yolonas"
            / ("python.exe" if os.name == "nt" else "bin/python")
        )

    def get_detector(self, name: str) -> BaseDetector:
        if name not in self.detectors:
            raise ValueError(f"Bilinmeyen tespit modeli: {name}")
        with self._lock:
            if self._active_detector and self._active_detector[0] == name:
                return self._active_detector[1]
            self._release_detector()
            config = self.detectors[name]
            try:
                detector = self._create_detector(config)
                self._active_detector = (name, detector)
                self._errors.pop(f"detector:{name}", None)
                return detector
            except Exception as exc:
                self._errors[f"detector:{name}"] = str(exc)
                self._clear_cuda()
                raise

    def get_classifier(
        self,
        name: str,
    ) -> PanelClassifier | PanelEnsembleClassifier:
        if name not in self.classifiers:
            raise ValueError(f"Bilinmeyen siniflandirma modeli: {name}")
        with self._lock:
            if self._active_classifier and self._active_classifier[0] == name:
                return self._active_classifier[1]
            self._release_classifier()
            config = self.classifiers[name]
            try:
                if config.loader == "ensemble":
                    members = [
                        self._create_classifier(self.classifiers[member_name])
                        for member_name in config.members
                    ]
                    classifier = PanelEnsembleClassifier(members, config.weights)
                else:
                    classifier = self._create_classifier(config)
                self._active_classifier = (name, classifier)
                self._errors.pop(f"classifier:{name}", None)
                return classifier
            except Exception as exc:
                self._errors[f"classifier:{name}"] = str(exc)
                self._clear_cuda()
                raise

    def _create_classifier(self, config: ClassifierConfig) -> PanelClassifier:
        return PanelClassifier(
            config.checkpoint_path,
            self.device,
            loader=config.loader,
            project_root=self.project_root,
            image_size=config.image_size,
        )

    def _create_detector(self, config: DetectorConfig) -> BaseDetector:
        if config.loader == "rfdetr":
            return RFDetrDetector(config.checkpoint_path, self.device)
        if config.loader == "ultralytics":
            return YoloDetector(config.checkpoint_path, self.device)
        if config.loader == "efficientdet":
            return EfficientDetDetector(
                config.checkpoint_path,
                config.model_name or "",
                config.image_size or 640,
                self.device,
            )
        if config.loader == "deformable_detr":
            return DeformableDetrDetector(config.checkpoint_path, self.device)
        if config.loader == "grounding_dino":
            return GroundingDinoDetector(config.checkpoint_path, self.device)
        if config.loader == "yolonas":
            python_path = self._yolonas_python_path()
            if not python_path.exists():
                raise FileNotFoundError(
                    "YOLO-NAS Python ortami bulunamadi. YOLO_NAS_PYTHON ayarlayin."
                )
            return YoloNasDetector(
                config.checkpoint_path,
                self.project_root,
                python_path,
            )
        raise ValueError(f"Desteklenmeyen tespit yukleyicisi: {config.loader}")

    def _release_detector(self) -> None:
        if self._active_detector is None:
            return
        _, detector = self._active_detector
        self._active_detector = None
        detector.close()
        del detector
        self._clear_cuda()

    def _release_classifier(self) -> None:
        if self._active_classifier is None:
            return
        _, classifier = self._active_classifier
        self._active_classifier = None
        classifier.close()
        del classifier
        self._clear_cuda()

    @staticmethod
    def _clear_cuda() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def warmup(self) -> None:
        for model_type, name in (("classifier", "edgenext"), ("detector", "yolov8s")):
            try:
                if model_type == "classifier":
                    self.get_classifier(name)
                else:
                    self.get_detector(name)
            except Exception:
                continue

    def _dependency_installed(self, loader: str) -> bool:
        dependencies = {
            "rfdetr": "rfdetr",
            "ultralytics": "ultralytics",
            "efficientdet": "effdet",
            "deformable_detr": "transformers",
            "grounding_dino": "transformers",
            "timm": "timm",
        }
        if loader == "yolonas":
            return self._yolonas_python_path().exists()
        if loader == "mambavision":
            return (
                self.project_root
                / "external/MambaVision/mambavision/models/mamba_vision.py"
            ).exists()
        if loader == "ensemble":
            return True
        if loader == "legacy_clip":
            return importlib.util.find_spec("open_clip") is not None
        if loader.startswith("legacy_dinov2"):
            return (
                Path.home()
                / ".cache/torch/hub/facebookresearch_dinov2_main"
            ).exists()
        if loader.startswith("legacy_"):
            return importlib.util.find_spec("torchvision") is not None
        return importlib.util.find_spec(dependencies[loader]) is not None

    def status(self) -> dict:
        active_detector = self._active_detector[0] if self._active_detector else None
        active_classifier = (
            self._active_classifier[0] if self._active_classifier else None
        )
        return {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "active_detector": active_detector,
            "active_classifier": active_classifier,
            "detectors": {
                name: {
                    **asdict(config),
                    "checkpoint_path": str(config.checkpoint_path),
                    "checkpoint_exists": config.checkpoint_path.exists(),
                    "dependency_installed": self._dependency_installed(config.loader),
                    "loaded": name == active_detector,
                    "error": self._errors.get(f"detector:{name}"),
                }
                for name, config in self.detectors.items()
            },
            "classifiers": {
                name: {
                    **asdict(config),
                    "checkpoint_path": str(config.checkpoint_path),
                    "checkpoint_exists": (
                        all(
                            self.classifiers[member].checkpoint_path.exists()
                            for member in config.members
                        )
                        if config.loader == "ensemble"
                        else config.checkpoint_path.exists()
                    ),
                    "dependency_installed": (
                        all(
                            self._dependency_installed(
                                self.classifiers[member].loader
                            )
                            for member in config.members
                        )
                        if config.loader == "ensemble"
                        else self._dependency_installed(config.loader)
                    ),
                    "loaded": name == active_classifier,
                    "error": self._errors.get(f"classifier:{name}"),
                }
                for name, config in self.classifiers.items()
            },
        }


class InferencePipeline:
    def __init__(self, project_root: Path, output_root: Path) -> None:
        self.project_root = project_root
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.models = ModelManager(project_root)

    def predict(
        self,
        image: Image.Image,
        filename: str,
        detector_model: str,
        classifier_model: str,
        mode: InferenceMode,
        threshold: float,
    ) -> dict:
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        request_dir = self.output_root / request_id
        request_dir.mkdir(parents=True, exist_ok=False)

        detection_started = time.perf_counter()
        if mode == "classification":
            detections = [Detection((0, 0, image.width, image.height), 1.0)]
        else:
            detections = self.models.get_detector(detector_model).predict(
                image,
                threshold,
            )
            detections = self._sanitize_detections(detections, image.size)
        detection_seconds = time.perf_counter() - detection_started

        if not detections:
            result = self._empty_result(
                request_id,
                filename,
                detector_model,
                classifier_model,
                mode,
                threshold,
                detection_seconds,
                time.perf_counter() - started,
            )
            image.save(request_dir / "annotated.jpg", format="JPEG", quality=92)
            self._write_result(request_dir, result)
            self._cleanup_old_outputs()
            return result

        padded_boxes = [self._padded_box(item.box, image.size) for item in detections]
        crops = [image.crop(box) for box in padded_boxes]
        classification_seconds = 0.0
        classifications = []
        if mode != "detection":
            classification_started = time.perf_counter()
            classifications = self.models.get_classifier(classifier_model).predict(crops)
            classification_seconds = time.perf_counter() - classification_started

        panel_results = []
        for index, (detection, box, crop) in enumerate(
            zip(detections, padded_boxes, crops),
            start=1,
        ):
            classification = (
                classifications[index - 1] if mode != "detection" else None
            )
            crop_name = f"panel_{index}.jpg"
            crop.save(request_dir / crop_name, format="JPEG", quality=92)
            panel_results.append(
                PanelResult(
                    panel_id=index,
                    box=box,
                    detection_confidence=(
                        None if mode == "classification" else detection.confidence
                    ),
                    class_name=(
                        classification.class_name if classification is not None else None
                    ),
                    class_label=(
                        CLASS_LABELS_TR[classification.class_name]
                        if classification is not None
                        else None
                    ),
                    classification_confidence=(
                        classification.confidence
                        if classification is not None
                        else None
                    ),
                    low_confidence=(
                        classification.confidence < 0.60
                        if classification is not None
                        else False
                    ),
                    probabilities=(
                        classification.probabilities
                        if classification is not None
                        else {}
                    ),
                    crop_url=f"/outputs/{request_id}/{crop_name}",
                )
            )

        self._annotate(image, panel_results).save(
            request_dir / "annotated.jpg",
            format="JPEG",
            quality=92,
        )
        result = self._build_result(
            request_id,
            filename,
            detector_model,
            classifier_model,
            mode,
            threshold,
            panel_results,
            detection_seconds,
            classification_seconds,
            time.perf_counter() - started,
        )
        self._write_result(request_dir, result)
        self._cleanup_old_outputs()
        return result

    def _build_result(
        self,
        request_id: str,
        filename: str,
        detector_model: str,
        classifier_model: str,
        mode: InferenceMode,
        threshold: float,
        panels: list[PanelResult],
        detection_seconds: float,
        classification_seconds: float,
        total_seconds: float,
    ) -> dict:
        detector = self.models.detectors.get(detector_model)
        classifier = self.models.classifiers.get(classifier_model)
        return {
            "request_id": request_id,
            "filename": filename,
            "mode": mode,
            "threshold": threshold,
            "device": self.models.device,
            "detector_model": detector_model if mode != "classification" else None,
            "detector_label": detector.label if detector and mode != "classification" else None,
            "classifier_model": classifier_model if mode != "detection" else None,
            "classifier_label": classifier.label if classifier and mode != "detection" else None,
            "panel_count": len(panels),
            "message": self._message(mode, len(panels)),
            "panels": [panel.to_dict() for panel in panels],
            "timings": {
                "detection_seconds": round(detection_seconds, 4),
                "classification_seconds": round(classification_seconds, 4),
                "total_seconds": round(total_seconds, 4),
            },
            "annotated_image_url": f"/outputs/{request_id}/annotated.jpg",
            "json_url": f"/outputs/{request_id}/result.json",
        }

    def _empty_result(
        self,
        request_id: str,
        filename: str,
        detector_model: str,
        classifier_model: str,
        mode: InferenceMode,
        threshold: float,
        detection_seconds: float,
        total_seconds: float,
    ) -> dict:
        return self._build_result(
            request_id,
            filename,
            detector_model,
            classifier_model,
            mode,
            threshold,
            [],
            detection_seconds,
            0.0,
            total_seconds,
        )

    @staticmethod
    def _message(mode: InferenceMode, panel_count: int) -> str:
        if panel_count == 0:
            return "Fotografta guven esigini gecen panel bulunamadi."
        if mode == "detection":
            return f"{panel_count} panel tespit edildi."
        if mode == "classification":
            return "Fotograf siniflandirildi."
        return f"{panel_count} panel tespit edildi ve siniflandirildi."

    @staticmethod
    def _sanitize_detections(
        detections: list[Detection], image_size: tuple[int, int]
    ) -> list[Detection]:
        width, height = image_size
        sanitized = []
        for item in detections:
            x1, y1, x2, y2 = item.box
            box = (
                max(0, min(x1, width - 1)),
                max(0, min(y1, height - 1)),
                max(1, min(x2, width)),
                max(1, min(y2, height)),
            )
            if box[2] - box[0] >= 8 and box[3] - box[1] >= 8:
                sanitized.append(Detection(box=box, confidence=item.confidence))
        return sanitized

    @staticmethod
    def _padded_box(
        box: tuple[int, int, int, int], image_size: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        width, height = image_size
        x1, y1, x2, y2 = box
        padding_x = max(2, int((x2 - x1) * 0.03))
        padding_y = max(2, int((y2 - y1) * 0.03))
        return (
            max(0, x1 - padding_x),
            max(0, y1 - padding_y),
            min(width, x2 + padding_x),
            min(height, y2 + padding_y),
        )

    @staticmethod
    def _font(size: int) -> ImageFont.ImageFont:
        for path in (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ):
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
        return ImageFont.load_default()

    def _annotate(self, image: Image.Image, panels: list[PanelResult]) -> Image.Image:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        line_width = max(3, round(min(image.size) / 240))
        font = self._font(max(15, round(min(image.size) / 35)))
        for panel in panels:
            color = "#F59E0B" if panel.low_confidence else "#10B981"
            draw.rectangle(panel.box, outline=color, width=line_width)
            if panel.classification_confidence is None:
                label = (
                    f"#{panel.panel_id} Panel "
                    f"%{(panel.detection_confidence or 0) * 100:.1f}"
                )
            else:
                label = (
                    f"#{panel.panel_id} {panel.class_label} "
                    f"%{panel.classification_confidence * 100:.1f}"
                )
            text_box = draw.textbbox((0, 0), label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            x1, y1, _, _ = panel.box
            label_top = max(0, y1 - text_height - 12)
            draw.rounded_rectangle(
                (x1, label_top, x1 + text_width + 16, label_top + text_height + 10),
                radius=5,
                fill=color,
            )
            draw.text((x1 + 8, label_top + 3), label, fill="white", font=font)
        return annotated

    @staticmethod
    def _write_result(request_dir: Path, result: dict) -> None:
        (request_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _cleanup_old_outputs(self) -> None:
        max_runs = int(os.getenv("DEMO_MAX_OUTPUTS", "50"))
        directories = sorted(
            (path for path in self.output_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for old_dir in directories[max_runs:]:
            shutil.rmtree(old_dir, ignore_errors=True)
