from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps, UnidentifiedImageError

from demo_web.inference.pipeline import InferencePipeline


logger = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
OUTPUT_ROOT = APP_DIR / "outputs"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

DETECTOR_SPEED_LABELS = {
    "rfdetr_small": "Orta",
    "deformable_detr": "Yavaş",
    "yolonas_s": "Orta",
    "efficientdet_d2": "Orta",
    "yolov8s": "Hızlı",
    "rtdetr_l": "Yavaş",
    "yolo11s": "Hızlı",
    "efficientdet_d1": "Orta",
    "yolo12s": "Hızlı",
    "grounding_dino_tiny": "Yavaş",
}

CLASSIFIER_SPEED_LABELS = {
    "focalnet": "Orta",
    "edgenext": "Hızlı",
    "mambavision": "Orta",
    "fastvit": "Hızlı",
    "repvit": "Hızlı",
    "tinyvit": "Hızlı",
    "maxvit": "Orta",
    "poolformer": "Hızlı",
    "coatnet": "Orta",
    "siglip": "Orta",
    "legacy_effnet_b3": "Hızlı",
    "legacy_convnext": "Orta",
    "legacy_swinv2": "Yavaş",
    "legacy_clip": "Yavaş",
    "legacy_maxvit": "Orta",
    "legacy_dinov2b": "Yavaş",
    "legacy_effnetv2l": "Yavaş",
    "legacy_vitlarge": "Yavaş",
    "legacy_dinov2s": "Orta",
    "ensemble_effnet_swin": "Yavaş",
}

DETECTOR_RECOMMENDATIONS = {
    "rfdetr_small": "En güçlü",
    "yolov8s": "Pratik",
}

CLASSIFIER_RECOMMENDATIONS = {
    "focalnet": "En güçlü",
    "edgenext": "Pratik",
}


def _read_image(data: bytes) -> Image.Image:
    if not data:
        raise HTTPException(status_code=400, detail="Bos dosya yuklenemez.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Dosya boyutu 15 MB sinirini asiyor.")
    try:
        image = Image.open(io.BytesIO(data))
        if image.format not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=415,
                detail="Yalnizca JPG, PNG ve WebP fotograflari desteklenir.",
            )
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=413,
                detail="Goruntu en fazla 32 megapiksel olabilir.",
            )
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=415, detail="Gecerli bir fotograf okunamadi.") from exc


def create_app(pipeline: InferencePipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if os.getenv("DEMO_EAGER_LOAD", "0") == "1":
            await run_in_threadpool(app.state.pipeline.models.warmup)
        yield

    app = FastAPI(
        title="Gunes Paneli Analiz Demo",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.pipeline = pipeline or InferencePipeline(PROJECT_ROOT, OUTPUT_ROOT)
    app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=OUTPUT_ROOT), name="outputs")
    templates = Jinja2Templates(directory=APP_DIR / "templates")

    @app.get("/", include_in_schema=False)
    async def index(request: Request):
        models = request.app.state.pipeline.models
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "detectors": models.detectors,
                "classifiers": models.classifiers,
                "detector_speed_labels": DETECTOR_SPEED_LABELS,
                "classifier_speed_labels": CLASSIFIER_SPEED_LABELS,
                "detector_recommendations": DETECTOR_RECOMMENDATIONS,
                "classifier_recommendations": CLASSIFIER_RECOMMENDATIONS,
                "detector_rankings": sorted(
                    models.detectors.items(),
                    key=lambda item: item[1].map50_95,
                    reverse=True,
                ),
                "classifier_rankings": sorted(
                    models.classifiers.items(),
                    key=lambda item: item[1].macro_f1,
                    reverse=True,
                ),
            },
        )

    @app.get("/api/health")
    async def health(request: Request):
        return {
            "status": "ok",
            "models": request.app.state.pipeline.models.status(),
            "limits": {
                "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
                "max_image_megapixels": MAX_IMAGE_PIXELS // 1_000_000,
            },
        }

    @app.post("/api/predict")
    async def predict(
        request: Request,
        file: UploadFile = File(...),
        detector_model: str = Form("yolov8s"),
        classifier_model: str = Form("edgenext"),
        mode: str = Form("full"),
        threshold: float = Form(0.25),
    ):
        models = request.app.state.pipeline.models
        if mode not in {"full", "classification", "detection"}:
            raise HTTPException(status_code=422, detail="Gecersiz tahmin modu.")
        if mode != "classification" and detector_model not in models.detectors:
            raise HTTPException(status_code=422, detail="Gecersiz tespit modeli.")
        if mode != "detection" and classifier_model not in models.classifiers:
            raise HTTPException(status_code=422, detail="Gecersiz siniflandirma modeli.")
        if not 0.05 <= threshold <= 0.95:
            raise HTTPException(
                status_code=422,
                detail="Tespit esigi 0.05 ile 0.95 arasinda olmalidir.",
            )

        data = await file.read(MAX_UPLOAD_BYTES + 1)
        image = await run_in_threadpool(_read_image, data)
        try:
            return await run_in_threadpool(
                request.app.state.pipeline.predict,
                image,
                file.filename or "fotograf",
                detector_model,
                classifier_model,
                mode,
                threshold,
            )
        except (RuntimeError, FileNotFoundError) as exc:
            logger.exception("Model kullanima hazirlanirken hata olustu.")
            raise HTTPException(
                status_code=503,
                detail="Secilen model su anda kullanilamiyor.",
            ) from exc
        except ValueError as exc:
            logger.exception("Tahmin girdisi islenirken hata olustu.")
            raise HTTPException(
                status_code=422,
                detail="Tahmin girdisi islenemedi.",
            ) from exc
        except Exception as exc:
            logger.exception("Tahmin sirasinda beklenmeyen hata olustu.")
            raise HTTPException(
                status_code=500,
                detail="Tahmin sirasinda beklenmeyen bir hata olustu.",
            ) from exc

    @app.get("/api/download/{request_id}/{filename}")
    async def download(request_id: str, filename: str):
        allowed = {"annotated.jpg", "result.json"}
        if filename not in allowed or not request_id.isalnum():
            raise HTTPException(status_code=404, detail="Dosya bulunamadi.")
        path = OUTPUT_ROOT / request_id / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="Dosya bulunamadi.")
        media_type = "image/jpeg" if filename.endswith(".jpg") else "application/json"
        return FileResponse(path, media_type=media_type, filename=filename)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("demo_web.app:app", host="127.0.0.1", port=8000, reload=False)
