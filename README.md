# Solar Panel Condition Analysis

End-to-end solar panel detection and condition classification system with
multiple trained deep learning models, ensemble inference, lazy model loading
and a local FastAPI web interface.

The pipeline first detects solar panels in an RGB image, crops each detected
panel region, and then classifies the panel condition.

```text
input image
  -> detection model: find solar panel bounding boxes
  -> panel crops
  -> classification model: clean / dust / snow / bird_drop / crack_or_damage
```

## Features

- 10 selectable solar panel detection models
- 19 selectable condition classification models
- EfficientNet-B3 + SwinV2-B weighted ensemble option
- Full pipeline, classification-only and detection-only modes
- Lazy loading: only the active detector and classifier stay in memory
- Responsive local web interface
- Annotated image and JSON result export
- Automated tests for the web API and inference pipeline

The web interface provides 29 selectable trained models in total: 10 detection
models and 19 classification models. The weighted ensemble is an additional
inference option, not a separately trained model. The downloadable archive
contains 33 trained-model files, including experimental models that are not
listed as separate selections in the web interface.

## Recommended Use

| Scenario | Detection model | Classification model | Reason |
|---|---|---|---|
| Best overall accuracy | RF-DETR Small | FocalNet | Strongest detection and classification results |
| Practical local demo | YOLOv8s | EdgeNeXt | Smaller and easier to deploy with strong results |
| Research comparison | Any selectable detector | Any selectable classifier | Useful for comparing architecture families |

For most users who only want to try the demo locally, start with:

```text
YOLOv8s detection + EdgeNeXt classification
```

For the strongest reported pipeline, use:

```text
RF-DETR Small detection + FocalNet classification
```

## Classification Classes

| Class | Meaning |
|---|---|
| `clean` | Clean panel |
| `dust` | Dusty or dirty panel |
| `snow` | Snow-covered panel |
| `bird_drop` | Panel with bird droppings |
| `crack_or_damage` | Cracked or physically damaged panel |

## System Requirements

### Minimum for Web Demo

| Component | Requirement |
|---|---|
| OS | Windows 10/11, Linux, or macOS |
| Python | 3.11 for the main demo environment |
| RAM | 8 GB minimum |
| Disk | Repository plus the downloaded model archive |
| GPU | Optional; CPU inference works but can be slow |

### Recommended

| Component | Recommendation |
|---|---|
| RAM | 16 GB or more |
| GPU | NVIDIA GPU with CUDA support for faster inference |
| Python environment | Conda or a clean virtual environment |
| Disk | Several GB free for trained model files and generated outputs |

Training all models requires substantially more compute than running the web
demo. The reported training runs used CUDA-capable NVIDIA GPUs. CPU-only use is
mainly practical for tests, the web app shell, and lightweight inference.

YOLO-NAS requires a separate Python 3.10 environment because of its dependency
constraints.

## Model Files

The trained model archive is hosted on Google Drive because it exceeds GitHub's
file size limits.

**Download:** [Google Drive model archive](https://drive.google.com/file/d/1dX6k0gjINOxWBufQCYlpZl74DiXqTd0h/view?usp=sharing)

Archive name:

```text
gunes_paneli_tum_modeller.zip
```

After downloading, extract the archive into the repository root. The following
directories should then exist:

```text
experiments/
legacy_models/
```

The archive checksum is listed in `SHA256SUMS.txt`.

## Model Size and Performance Summary

| Model | Task | Size | Main metric | Note |
|---|---|---:|---:|---|
| RF-DETR Small | Detection | 121.47 MB | mAP50-95 0.7524 | Strongest detector |
| YOLOv8s | Detection | 21.48 MB | mAP50-95 0.6093 | Practical detector |
| EfficientDet-D2 | Detection | 32.89 MB | mAP50-95 0.6094 | Balanced alternative |
| FocalNet | Classification | 105.60 MB | Macro F1 0.9937 | Strongest classifier |
| EdgeNeXt | Classification | 8.25 MB | Macro F1 0.9906 | Best lightweight classifier |
| MambaVision | Classification | 118.94 MB | Macro F1 0.9902 | Strong research baseline |

See [docs/model_comparison.md](docs/model_comparison.md) for the fuller model
comparison tables.

## Installation

Python 3.11 is recommended for the main demo environment.

### Conda

```powershell
conda env create -f environment.yml
conda activate solar-panel-cls
```

### Python venv

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
```

On Windows PowerShell, activate the venv with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### YOLO-NAS Optional Environment

YOLO-NAS uses a separate Python 3.10 environment:

```powershell
conda create -n solar-panel-yolonas python=3.10 -y
conda activate solar-panel-yolonas
python -m pip install super-gradients==3.7.1
```

Set its interpreter path when it is not located at the default conda path:

```powershell
$env:YOLO_NAS_PYTHON = "C:\path\to\solar-panel-yolonas\python.exe"
```

## Run

Windows:

```powershell
.\scripts\run_web_demo.ps1
```

Linux or macOS:

```bash
python -m uvicorn demo_web.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

API documentation is available at `http://127.0.0.1:8000/docs`.

DINOv2 models may require internet access on first use when their Torch Hub
resources are not already cached. CLIP ViT-L/14 requires `open_clip_torch`.

## Repository Layout

| Path | Purpose |
|---|---|
| `demo_web/` | FastAPI app, frontend, and inference pipeline |
| `scripts/` | Dataset preparation, training, evaluation, and demo helpers |
| `tests/` | Automated tests for the web demo and pipeline behavior |
| `docs/` | Additional documentation and model comparison tables |
| `reports/` | Training summaries and evaluation reports |
| `external/MambaVision/` | Third-party NVIDIA MambaVision source files |
| `experiments/` | Downloaded trained model files, ignored by Git |
| `legacy_models/` | Downloaded legacy model files, ignored by Git |

## Tests

```powershell
python -m pytest tests/test_demo_web.py -q
```

GitHub Actions runs the test suite on every push and pull request. The tests use
fake lightweight models, so they verify application behavior without requiring
the full trained model archive.

## Notes on Scores

Some classification scores originate from different experimental datasets and
training phases. They are displayed as reported results and should not be
treated as a perfectly controlled benchmark across every model.

For detection, `mAP50-95` is the main comparison metric because it is stricter
than `mAP50` and better reflects bounding-box quality.

## Third-Party Code and License

Original project code is licensed under the [MIT License](LICENSE):

```text
Copyright (c) 2026 Yusuf Utku Ozturk
```

The MIT license applies only to original project code. It does not relicense
bundled third-party code, trained model files, pretrained weights, datasets, or
sample images.

The bundled MambaVision source files remain under the NVIDIA Source Code
License-NC, which limits covered code to non-commercial research or evaluation
use. MambaVision pretrained weights are subject to CC BY-NC-SA 4.0.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the retained
`external/MambaVision/LICENSE` file for details.
