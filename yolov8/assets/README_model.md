---
tags:
  - yolov8
  - segmentation
  - ultralytics
---

# YOLOv8 Segmentation Model

- **Task:** Instance segmentation (masks + boxes)
- **Base weights:** yolov8n-seg.pt (COCO pretrained)
- **Dataset:** [your dataset name]
- **Classes:** [class0, class1, ...]
- **Input size (imgsz):** [e.g., 640]
- **Training:** epochs=[..], batch=[..], device=[..]

## How to use

### CLI
```bash
yolo segment predict model=best.pt source=path/to/images imgsz=640 conf=0.25
```

### Python
```python
from ultralytics import YOLO
model = YOLO("best.pt")
results = model("path/to/image.jpg", imgsz=640, conf=0.25)
```
