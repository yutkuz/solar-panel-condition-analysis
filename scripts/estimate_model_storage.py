from __future__ import annotations

import json
from pathlib import Path

from PIL import Image  # noqa: F401
import timm


TIMM_MODELS = {
    "fastvit": "fastvit_t8",
    "repvit": "repvit_m0_9",
    "tinyvit": "tiny_vit_5m_224",
    "poolformer": "poolformer_s12",
    "edgenext": "edgenext_x_small",
    "focalnet": "focalnet_tiny_srf",
    "maxvit": "maxvit_tiny_rw_224",
    "coatnet": "coatnet_0_rw_224",
    "siglip": "vit_base_patch16_siglip_224",
    "clip": "vit_base_patch16_clip_224",
}

EXTERNAL_PARAM_M = {
    "mambavision": 31.8,
    "vmamba": 30.0,
    "hornet": 22.4,
}


def checkpoint_mb(param_m: float) -> float:
    return param_m * 4.0


def main() -> None:
    rows = []
    for key, model_name in TIMM_MODELS.items():
        model = timm.create_model(model_name, pretrained=False, num_classes=5)
        params = sum(p.numel() for p in model.parameters()) / 1_000_000
        rows.append(
            {
                "model": key,
                "params_m": round(params, 2),
                "one_fp32_checkpoint_mb": round(checkpoint_mb(params), 1),
            }
        )
    for key, params in EXTERNAL_PARAM_M.items():
        rows.append(
            {
                "model": key,
                "params_m": params,
                "one_fp32_checkpoint_mb": round(checkpoint_mb(params), 1),
            }
        )

    total_one = sum(r["one_fp32_checkpoint_mb"] for r in rows)
    # Rough estimate for final, tuning, and cached pretrained weights.
    estimate = {
        "models": rows,
        "total_one_checkpoint_mb": round(total_one, 1),
        "estimated_project_outputs_mb": round(total_one * 2.2, 1),
        "estimated_with_pretrained_cache_mb": round(total_one * 3.2, 1),
    }
    out = Path("experiments/classification/storage_estimate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(estimate, indent=2), encoding="utf-8")
    print(json.dumps(estimate, indent=2))


if __name__ == "__main__":
    main()
