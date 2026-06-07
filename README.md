# Solar Panel Condition Analysis

End-to-end solar panel detection and condition classification system with
multiple deep learning architectures, ensemble inference, lazy model loading
and a FastAPI web interface.

## Features

- 10 selectable solar panel detection models
- 19 selectable condition classification models
- EfficientNet-B3 + SwinV2-B weighted ensemble
- Full pipeline, classification-only and detection-only modes
- Lazy loading: only the active detector and classifier stay in memory
- Responsive local web interface
- Annotated image and JSON result export

The web interface provides 29 selectable trained models in total: 10 detection
models and 19 classification models. The weighted ensemble is an additional
inference option, not a separately trained model. The downloadable archive
contains 33 trained-model files, including experimental models that are not
listed as separate selections in the web interface.

## Classification Classes

- Clean
- Dust
- Snow
- Bird drop
- Crack or physical damage

## Model Files

The model archive is hosted on Google Drive because it exceeds GitHub's file
size limits.

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

## Installation

Python 3.11 is recommended.

```powershell
conda env create -f environment.yml
conda activate solar-panel-cls
```

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

## Tests

```powershell
python -m pytest tests/test_demo_web.py -q
```

GitHub Actions runs the test suite on every push and pull request.

## Documentation

Additional architecture and demo documentation is available in `docs/`.

## Model Score Note

Some classification scores originate from different experimental datasets and
training phases. They are displayed as reported results and should not be
treated as a perfectly controlled benchmark across every model.

## License

Original project code is licensed under the [MIT License](LICENSE):

```text
Copyright (c) 2026 Yusuf Utku Öztürk
```

The MIT license applies only to original project code. It does not relicense
bundled third-party code, trained model files, pretrained weights, datasets, or
sample images.

The bundled MambaVision source files remain under the NVIDIA Source Code
License-NC, which limits covered code to non-commercial research or evaluation
use. MambaVision pretrained weights are subject to CC BY-NC-SA 4.0.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the retained
`external/MambaVision/LICENSE` file for details.
