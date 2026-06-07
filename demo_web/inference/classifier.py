from __future__ import annotations

import sys
import threading
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn

from .schemas import Classification


LEGACY_CLASS_NAMES = {
    "bird-drop": "bird_drop",
    "clean": "clean",
    "dusty": "dust",
    "physical-damage": "crack_or_damage",
    "snow-covered": "snow",
}


class LegacyFeatureClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, inputs):
        features = self.backbone(inputs)
        if isinstance(features, tuple):
            features = features[0]
        return self.head(features.float())


class PanelClassifier:
    def __init__(
        self,
        checkpoint_path: Path,
        device: str,
        loader: str = "timm",
        project_root: Path | None = None,
        image_size: int = 224,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._lock = threading.Lock()

        try:
            import torch
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError(
                "Siniflandirma bagimliliklari eksik. requirements-demo.txt dosyasini kurun."
            ) from exc

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Siniflandirma checkpoint bulunamadi: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.classes = [
            LEGACY_CLASS_NAMES.get(name, name)
            for name in checkpoint["classes"]
        ]
        self.model_name = str(checkpoint.get("model_name", loader))
        if loader.startswith("legacy_"):
            self.model = self._build_legacy_model(
                loader,
                checkpoint,
                len(self.classes),
            )
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model_state"))
        elif loader == "mambavision":
            if project_root is None:
                raise RuntimeError("MambaVision icin proje kok dizini gerekli.")
            external_path = project_root / "external" / "MambaVision"
            if str(external_path) not in sys.path:
                sys.path.insert(0, str(external_path))
            from mambavision.models.mamba_vision import mamba_vision_T

            self.model = mamba_vision_T(
                pretrained=False,
                num_classes=len(self.classes),
            )
            state_dict = checkpoint["model"]
        else:
            from timm import create_model

            self.model = create_model(
                self.model_name,
                pretrained=False,
                num_classes=len(self.classes),
            )
            state_dict = checkpoint["model"]
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self._torch = torch

    @staticmethod
    def _build_legacy_model(loader: str, checkpoint: dict, num_classes: int) -> nn.Module:
        from torchvision.models import (
            convnext_base,
            efficientnet_b3,
            efficientnet_v2_l,
            maxvit_t,
            swin_v2_b,
            vit_l_16,
        )

        params = checkpoint.get("params", {})
        dropout = float(params.get("dropout", 0.0))
        if loader == "legacy_effnet_b3":
            model = efficientnet_b3(weights=None)
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features,
                num_classes,
            )
            return model
        if loader == "legacy_convnext":
            model = convnext_base(weights=None)
            model.classifier[2] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(model.classifier[2].in_features, num_classes),
            )
            return model
        if loader == "legacy_swinv2":
            model = swin_v2_b(weights=None)
            model.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(model.head.in_features, num_classes),
            )
            return model
        if loader == "legacy_maxvit":
            model = maxvit_t(weights=None)
            model.classifier[-1] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(model.classifier[-1].in_features, num_classes),
            )
            return model
        if loader == "legacy_effnetv2l":
            model = efficientnet_v2_l(weights=None)
            model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(model.classifier[1].in_features, num_classes),
            )
            return model
        if loader == "legacy_vitlarge":
            model = vit_l_16(weights=None)
            model.heads = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(model.heads.head.in_features, num_classes),
            )
            return model
        if loader == "legacy_clip":
            try:
                import open_clip
            except ImportError as exc:
                raise RuntimeError("Eski CLIP modeli icin open_clip eksik.") from exc
            clip_model = open_clip.create_model("ViT-L-14", pretrained=None).visual
            return LegacyFeatureClassifier(
                clip_model,
                clip_model.output_dim,
                num_classes,
                dropout,
            )
        if loader in {"legacy_dinov2s", "legacy_dinov2b"}:
            model_name = (
                "dinov2_vits14" if loader == "legacy_dinov2s" else "dinov2_vitb14"
            )
            backbone = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                pretrained=False,
                verbose=False,
            )
            return LegacyFeatureClassifier(
                backbone,
                backbone.embed_dim,
                num_classes,
                dropout,
            )
        raise ValueError(f"Desteklenmeyen eski siniflandirici: {loader}")

    def predict(self, images: list[Image.Image]) -> list[Classification]:
        if not images:
            return []

        batch = self._torch.stack([self.transform(image.convert("RGB")) for image in images])
        batch = batch.to(self.device)
        with self._lock, self._torch.inference_mode():
            logits = self.model(batch)
            if isinstance(logits, dict):
                logits = logits["logits"]
            probabilities = self._torch.softmax(logits, dim=1).detach().cpu()

        results = []
        for row in probabilities:
            index = int(row.argmax().item())
            class_name = self.classes[index]
            results.append(
                Classification(
                    class_name=class_name,
                    confidence=float(row[index].item()),
                    probabilities={
                        name: float(row[class_index].item())
                        for class_index, name in enumerate(self.classes)
                    },
                )
            )
        return results

    def close(self) -> None:
        self.model.to("cpu")


class PanelEnsembleClassifier:
    def __init__(
        self,
        classifiers: list[PanelClassifier],
        weights: tuple[float, ...],
    ) -> None:
        if len(classifiers) != len(weights) or not classifiers:
            raise ValueError("Ensemble modelleri ve agirliklari uyusmuyor.")
        total = sum(weights)
        if total <= 0:
            raise ValueError("Ensemble agirlik toplami sifirdan buyuk olmali.")
        self.classifiers = classifiers
        self.weights = tuple(weight / total for weight in weights)

    def predict(self, images: list[Image.Image]) -> list[Classification]:
        member_results = [
            classifier.predict(images)
            for classifier in self.classifiers
        ]
        results = []
        for image_index in range(len(images)):
            probabilities: dict[str, float] = {}
            for weight, predictions in zip(self.weights, member_results):
                for class_name, probability in predictions[image_index].probabilities.items():
                    probabilities[class_name] = (
                        probabilities.get(class_name, 0.0) + weight * probability
                    )
            class_name = max(probabilities, key=probabilities.get)
            results.append(
                Classification(
                    class_name=class_name,
                    confidence=probabilities[class_name],
                    probabilities=probabilities,
                )
            )
        return results

    def close(self) -> None:
        for classifier in self.classifiers:
            classifier.close()
