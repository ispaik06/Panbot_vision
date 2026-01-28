# Panbot_vision

**Panbot_vision** is a vision pipeline repository for the Panbot (pancake-cooking robot) project.

The primary goal is to **detect and measure batter coverage on a pan** reliably, and to support an end-to-end workflow from **data collection → annotation → training → live inference**.

This repo includes:
- **Calibration + warp (top-view normalization)** using corner points
- **SAM-based segmentation labeling** to quickly create masks
- **YOLOv8 segmentation** training + real-time inference to estimate batter area
- **(Optional) ResNet18 + GRU sequence model** for temporal decision/classification from frame sequences

---

## Directory Overview

```text
Panbot_vision/
├── calibration/                # Corner JSONs for warp/top-view transform
│   ├── corners_1080p.json
│   └── corners.json
├── environment.yml             # Conda environment (exported from "sam" env)
├── images/                     # Labeled/curated images (often ignored in git)
├── masks/                      # Mask images aligned with images/ (often ignored in git)
├── model/                      # Model checkpoints (SAM, etc.)
│   └── sam_vit_b_01ec64.pth
├── raw_datasets/               # Raw and warped image collections
│   ├── raw_images/
│   └── warped_images/
├── resnet18_gru16/             # Temporal model pipeline (ResNet18 + GRU)
├── scripts/                    # Main scripts: calibration/warp/labeling/capture
├── tools/                      # Helper utilities: rename, merge, warp batch, etc.
├── videos/                     # Recorded videos (optional / local)
├── yolo26n.pt                  # (Optional) YOLO weight file (custom)
└── yolov8/                     # YOLOv8 segmentation workspace
```

---

## Environment Setup (Conda + pip)

This project is typically used with a Conda environment (commonly named `sam`).  
`environment.yml` is exported so others can reproduce the environment more easily.

### Create environment

```bash
cd ~/Panbot_vision
conda env create -f environment.yml
conda activate sam
```

### Update existing environment

```bash
conda activate sam
conda env update -f environment.yml --prune
```

> Note: `environment.yml` usually contains both conda dependencies and a `pip:` section for pip-installed packages.

---

## What you can do with SAM in this repo

SAM (Segment Anything Model) is used to speed up segmentation annotation.

Typical workflow:
1. Collect images (raw and/or warped)
2. Use SAM labeling tool to create segmentation masks
3. Convert masks → YOLOv8 segmentation labels
4. Train YOLOv8-seg
5. Run live inference and compute batter area

### Relevant files
- `model/sam_vit_b_01ec64.pth`: SAM ViT-B checkpoint used by labeling scripts
- `scripts/sam_labeler.py`: interactive SAM labeling tool (saves masks)

---

## Calibration & Warp (Top-view normalization)

Warping helps stabilize batter measurement by mapping the pan region into a consistent plane.

### Calibration files
- `calibration/corners.json`
- `calibration/corners_1080p.json`

These store corner points for perspective transform.

### Scripts
- `scripts/calibration.py`
- `scripts/calibrate_from_image.py`
- `scripts/show_corners_and_warp.py`
- `scripts/capture_warped_images.py`

---

## YOLOv8 segmentation (`yolov8/`)

The `yolov8/` directory is a self-contained workspace for training and inference using YOLOv8 segmentation.

```text
yolov8/
├── assets/
│   └── README_model.md         # Model notes (optional)
├── dataset/
│   ├── dataset.yaml            # Dataset config
│   ├── raw/                    # Raw images/masks (input)
│   └── yolo/                   # YOLO format dataset (output)
├── runs/
│   └── batter_seg_local_v1/    # Training outputs (weights/logs)
├── scripts/
│   ├── masks_to_yolo_seg.py    # Convert mask images → YOLO-seg labels
│   ├── train_seg.py            # Training wrapper
│   └── predict_batter_area.py  # Live inference + batter area measurement
└── weights/
    └── yolov8s-seg.pt          # Base pretrained segmentation weight
```

### Typical flow

#### A) Prepare dataset
- Put images into: `yolov8/dataset/raw/images`
- Put masks into: `yolov8/dataset/raw/masks`
- Convert masks → YOLO segmentation labels:

```bash
conda activate sam
python yolov8/scripts/masks_to_yolo_seg.py
```

#### B) Train
```bash
python yolov8/scripts/train_seg.py --dataset_yaml yolov8/dataset/dataset.yaml
```

#### C) Live inference + area measurement
```bash
python yolov8/scripts/predict_batter_area.py \
  --model yolov8/runs/batter_seg_local_v1/weights/best.pt \
  --cam 0
```

> The inference script typically supports options such as camera backend (v4l2), MJPG, resolution, warp corners, preview scale, etc.

---

## ResNet18 + GRU sequence model (`resnet18_gru16/`)

This folder contains a temporal model pipeline using:
- **ResNet18** as an image feature extractor
- **GRU (hidden size 16)** to model time dynamics across sequences

This can be used for sequence-based classification/decision tasks, e.g.:
- “batter coverage is sufficient”
- “spreading stabilized / ready state”
- “success state detection” from video segments

```text
resnet18_gru16/
├── assets/
│   └── annotations.csv         # Labels/metadata for runs
├── data_raw/                   # Raw run folders (often excluded)
├── dataset/
│   ├── index_train.csv
│   ├── index_val.csv
│   └── index_test.csv          # Split indices
├── scripts/
│   ├── train.py                # Train the sequence model
│   ├── eval.py                 # Evaluate trained checkpoints
│   └── infer_live.py           # Live inference from camera/stream
└── tools/
    ├── extract_frames_from_runs.py
    ├── make_sequence_index.py
    └── warp_runs_batch.py
```

---

## `scripts/` (main scripts)

- `save_raw_images.py`: save raw camera frames
- `capture_warped_images.py`: capture images after applying warp transform
- `record_video.py`: simple video recording utility
- `measure_batter_area_1.py`: legacy/experimental area measurement script
- `sam_labeler.py`: interactive SAM labeling tool
- `show_corners_and_warp.py`: visualize corners + warp result

---

## `tools/` (helper utilities)

- `warp_images_batch.py`: batch warp images using calibration corners
- `video_to_frames.py`: extract frames from videos
- `rename_dataset.py`: enforce consistent naming (e.g., `img_000001.jpg`)
- `merge_images_to_dataset.py`: merge multiple folders into one dataset
- `webcam_test.py`: quick camera connectivity test

---

## Naming Conventions

Common naming style:
- Image: `img_000001.jpg`
- Mask:  `img_000001.png`

Masks should align 1:1 with images (same stem).

---

## Notes / Disclaimer

This repository contains experimental code used for robotics research and prototyping.
Feel free to modify and extend it for your own pipeline.
