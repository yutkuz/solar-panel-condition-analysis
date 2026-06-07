# Uctan Uca Gunes Paneli Web Demo

Bu uygulama bir fotograf uzerinde panel tespiti ve panel durum
siniflandirmasini tek web arayuzunde calistirir.

## Model Katalogu

Arayuzde birbirinden bagimsiz secilebilen:

- 10 tespit modeli
- 19 siniflandirma modeli ve 1 ensemble secenegi
- Tam akis, sadece siniflandirma ve sadece tespit modlari

bulunur.

Ensemble secenegi EfficientNet-B3 ve SwinV2-B modellerinin sirasiyla
`0.571 / 0.429` agirlikli softmax olasilik ortalamasini kullanir.

Siniflandirma katalogu, guncel 12 modelden istenen ilk 10 modeli ve onceki
calismadaki 9 modeli kapsar. Guncel CLIP ve HorNet bu demo kataloguna
dahil edilmemistir.

Modeller lazy-load ile ihtiyac aninda yuklenir. Bellekte ayni anda en fazla
bir tespit ve bir siniflandirma modeli tutulur. Secim degistiginde ayni
turdeki onceki model bellekten cikarilir.

YOLO-NAS modeli ayri Python 3.10 ortaminda kalici bir worker olarak
calisir. Windows icin varsayilan yol:

```text
%USERPROFILE%\anaconda3\envs\solar-panel-yolonas\python.exe
```

Farkli bir ortam icin `YOLO_NAS_PYTHON` ortam degiskeni kullanilabilir.

## Kurulum

Python 3.11 ortami onerilir.

```powershell
conda activate solar-panel-cls
python -m pip install -r requirements-demo.txt
```

Guncel checkpointler:

```text
experiments/classification/<model>/final/best.pt
experiments/detection/<model>/final/...
```

Onceki classification checkpointleri:

```text
legacy_models/
```

## Calistirma

```powershell
.\scripts\run_web_demo.ps1
```

Ardindan `http://127.0.0.1:8000` adresini acin.

Varsayilan EdgeNeXt ve YOLOv8s modellerini sunucu acilisinda yuklemek icin:

```powershell
.\scripts\run_web_demo.ps1 -EagerLoad
```

API dokumani `http://127.0.0.1:8000/docs`, model durumu ise
`http://127.0.0.1:8000/api/health` adresindedir.

## Test

```powershell
python -m pytest tests/test_demo_web.py -q
```

## Harici Test Seti

```powershell
python scripts/collect_external_test_images.py
python scripts/evaluate_external_demo_set.py `
  --detector-model yolov8s `
  --classifier-model edgenext `
  --output reports/external_demo_edgenext_yolov8s.json
```

## Sinirlar

- Dosya boyutu en fazla 15 MB'dir.
- Goruntu en fazla 32 megapiksel olabilir.
- JPG, PNG ve WebP desteklenir.
- Siniflandirma guveni 0.60 altindaysa sonuc dusuk guvenli isaretlenir.
- Uretim ciktisi klasorunde en yeni 50 istek tutulur.
