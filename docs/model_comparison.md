# Model Comparison

This document summarizes the reported model size and performance results for
the solar panel condition analysis project. The README keeps only the short
decision table; this file keeps the detailed comparison.

Sizes refer to the saved final checkpoints used by the demo archive when a
single final checkpoint is available. Some model families may require extra
framework files or cached pretrained resources at runtime.

## Recommended Pipelines

| Scenario | Detection model | Classification model | Reason |
|---|---|---|---|
| Best overall accuracy | RF-DETR Small | FocalNet | Highest reported detection and classification results |
| Practical local demo | YOLOv8s | EdgeNeXt | Smaller checkpoints and easier deployment |
| Research-oriented comparison | RF-DETR Small or Deformable DETR | MambaVision or FocalNet | Stronger architectural comparison value |

## Detection Models

Main metric: `mAP50-95`.

| Rank | Model | mAP50-95 | mAP50 | Precision@IoU50 | Recall@IoU50 | F1@IoU50 | Final checkpoint size | Note |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | RF-DETR Small | 0.7524 | 0.9140 | 0.8862 | 0.8916 | 0.8889 | 121.47 MB | Strongest detector |
| 2 | Deformable DETR | 0.6232 | 0.8224 | 0.6667 | 0.8193 | 0.7351 | 153.23 MB | Transformer baseline |
| 3 | YOLO-NAS-S | 0.6119 | 0.8596 | 0.4261 | 0.9036 | 0.5792 | 244.31 MB | Highest recall, lower precision |
| 4 | EfficientDet-D2 | 0.6094 | 0.8395 | 0.8293 | 0.8193 | 0.8242 | 32.89 MB | Balanced alternative |
| 5 | YOLOv8s | 0.6093 | 0.8307 | 0.6832 | 0.8313 | 0.7500 | 21.48 MB | Practical YOLO option |
| 6 | RT-DETR-L | 0.6053 | 0.8257 | - | - | - | 63.15 MB | Ultralytics RT-DETR model |
| 7 | YOLOv11s | 0.6005 | 0.8267 | 0.7529 | 0.7892 | 0.7706 | 18.29 MB | Strong YOLO baseline |
| 8 | EfficientDet-D1 | 0.5950 | 0.8134 | 0.7904 | 0.7952 | 0.7928 | 26.78 MB | Smaller EfficientDet baseline |
| 9 | YOLOv12s | 0.5813 | 0.8391 | - | - | - | 18.06 MB | Good mAP50, weaker mAP50-95 |
| 10 | Grounding DINO Tiny | 0.5806 | 0.7561 | 0.8268 | 0.6325 | 0.7167 | 657.42 MB | Experimental prompt-based detector |

## Classification Models

Main metric: macro F1.

| Rank | Model | Test accuracy | Macro F1 | Macro precision | Macro recall | Best val acc | Sec/image | Final checkpoint size | Note |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | FocalNet | 0.9929 | 0.9937 | 0.9941 | 0.9935 | 0.9905 | 0.0324 | 105.60 MB | Strongest classifier |
| 2 | EdgeNeXt | 0.9905 | 0.9906 | 0.9913 | 0.9899 | 0.9905 | 0.0306 | 8.25 MB | Best lightweight classifier |
| 3 | MambaVision | 0.9905 | 0.9902 | 0.9906 | 0.9899 | 0.9881 | 0.0343 | 118.94 MB | Strong Mamba-family result |
| 4 | FastViT | 0.9834 | 0.9831 | 0.9832 | 0.9831 | 0.9929 | 0.0257 | 12.69 MB | Fast compact model |
| 5 | RepViT | 0.9786 | 0.9786 | 0.9790 | 0.9786 | 0.9929 | 0.0257 | 18.42 MB | Compact baseline |
| 6 | TinyViT | 0.9786 | 0.9781 | 0.9778 | 0.9790 | 0.9786 | 0.0253 | 19.48 MB | Compact baseline |
| 7 | MaxViT | 0.9762 | 0.9750 | 0.9771 | 0.9739 | 0.9952 | 0.0258 | 109.23 MB | Strong but larger |
| 8 | PoolFormer | 0.9691 | 0.9697 | 0.9730 | 0.9674 | 0.9881 | 0.0267 | 43.55 MB | Moderate size |
| 9 | CoAtNet | 0.9549 | 0.9564 | 0.9594 | 0.9548 | 0.9762 | 0.0255 | 101.86 MB | Larger baseline |
| 10 | SigLIP | 0.8646 | 0.8666 | 0.8792 | 0.8626 | 0.8765 | 0.0266 | 354.39 MB | Large checkpoint |
| 11 | CLIP fine-tuning | 0.7458 | 0.7524 | 0.7636 | 0.7457 | 0.7482 | 0.0268 | 327.36 MB | Lower result in this setup |
| 12 | HorNet | 0.6936 | 0.6994 | 0.7109 | 0.6955 | 0.7150 | 0.0395 | 83.68 MB | CPU/no-pretrained caveat |

## Interpretation

For classification, FocalNet gives the strongest macro F1. EdgeNeXt is the most
practical option because its checkpoint is much smaller while remaining very
close in score. MambaVision is useful as a research comparison, but the bundled
MambaVision source code and pretrained weights remain third-party NVIDIA
assets with separate license terms.

For detection, RF-DETR Small is the strongest model by mAP50-95 and balanced
IoU50 metrics. YOLOv8s and EfficientDet-D2 are more practical lightweight
alternatives. YOLO-NAS-S gives high recall but lower precision, which means it
can find more panels while also producing more false positives.

## Metric Notes

`mAP50` uses an IoU threshold of 0.50 and is more tolerant of loose boxes.
`mAP50-95` averages over stricter IoU thresholds from 0.50 to 0.95, so it is a
better general metric for bounding-box quality.

Macro F1 is used for classification because it treats each class more evenly
than plain accuracy when the class distribution is not perfectly balanced.
