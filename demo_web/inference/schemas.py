from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


InferenceMode = Literal["full", "classification", "detection"]


CLASS_LABELS_TR = {
    "bird_drop": "Kus pisligi",
    "clean": "Temiz",
    "crack_or_damage": "Catlak veya hasar",
    "dust": "Toz",
    "snow": "Kar",
}


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class Classification:
    class_name: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class PanelResult:
    panel_id: int
    box: tuple[int, int, int, int]
    detection_confidence: float | None
    class_name: str | None
    class_label: str | None
    classification_confidence: float | None
    low_confidence: bool
    probabilities: dict[str, float]
    crop_url: str

    def to_dict(self) -> dict:
        return asdict(self)
