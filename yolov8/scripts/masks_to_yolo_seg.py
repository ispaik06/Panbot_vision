import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def find_mask_for_image(image_path: Path, masks_dir: Path) -> Path | None:
    """이미지 stem과 동일한 mask 파일을 masks_dir에서 찾습니다."""
    stem = image_path.stem
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        cand = masks_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def binarize_mask(mask_any: np.ndarray, thr: int = 20) -> np.ndarray:
    """mask가 밝은 영역=반죽이라고 가정하고 이진화합니다."""
    if mask_any.ndim == 3:
        gray = cv2.cvtColor(mask_any, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_any
    _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    return bw


def contours_to_yolo_seg_lines(
    bw_mask: np.ndarray,
    class_id: int = 0,
    min_area_px: int = 200,
    approx_eps_ratio: float = 0.002,
):
    """
    YOLOv8 segmentation 라벨 포맷:
      class x1 y1 x2 y2 ... (0~1 정규화)
    """
    h, w = bw_mask.shape[:2]
    cnts, _ = cv2.findContours(bw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area_px:
            continue

        peri = cv2.arcLength(c, True)
        eps = approx_eps_ratio * peri
        approx = cv2.approxPolyDP(c, eps, True)  # (N,1,2)

        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            continue

        norm = []
        for x, y in pts:
            nx = float(x) / float(w)
            ny = float(y) / float(h)
            nx = min(max(nx, 0.0), 1.0)
            ny = min(max(ny, 0.0), 1.0)
            norm.extend([nx, ny])

        lines.append(str(class_id) + " " + " ".join(f"{v:.6f}" for v in norm))

    return lines


def write_dataset_yaml(dataset_dir: Path, names: list[str]):
    """
    dataset_dir/
      dataset.yaml
      yolo/images/train ...
    """
    data = {
        "path": "yolo",
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    (dataset_dir / "dataset.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--raw_images", type=str, default="yolov8/dataset/raw/images",
                    help="raw images dir (default matches your folder)")
    ap.add_argument("--raw_masks", type=str, default="yolov8/dataset/raw/masks",
                    help="raw masks dir (default matches your folder)")
    ap.add_argument("--dataset_dir", type=str, default="yolov8/dataset",
                    help="dataset root (contains yolo/ and dataset.yaml)")

    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--class_name", type=str, default="batter")
    ap.add_argument("--class_id", type=int, default=0)

    ap.add_argument("--mask_thr", type=int, default=20)
    ap.add_argument("--min_area_px", type=int, default=200)
    ap.add_argument("--approx_eps_ratio", type=float, default=0.002)

    ap.add_argument("--copy_images", action="store_true",
                    help="copy raw images into yolo/images (recommended)")
    args = ap.parse_args()

    # project root = PANBOT_VISION
    project_root = Path(__file__).resolve().parents[2]

    raw_images = (project_root / args.raw_images).resolve()
    raw_masks = (project_root / args.raw_masks).resolve()
    dataset_dir = (project_root / args.dataset_dir).resolve()

    assert raw_images.exists(), f"raw_images not found: {raw_images}"
    assert raw_masks.exists(), f"raw_masks not found: {raw_masks}"
    ensure_dir(dataset_dir)

    yolo_root = dataset_dir / "yolo"
    img_train = yolo_root / "images" / "train"
    img_val = yolo_root / "images" / "val"
    lab_train = yolo_root / "labels" / "train"
    lab_val = yolo_root / "labels" / "val"
    for p in [img_train, img_val, lab_train, lab_val]:
        ensure_dir(p)

    imgs = sorted([p for p in raw_images.iterdir() if p.is_file() and is_image(p)])
    pairs = []
    for im in imgs:
        m = find_mask_for_image(im, raw_masks)
        if m is not None:
            pairs.append((im, m))

    if not pairs:
        raise RuntimeError("No (image, mask) pairs found. Check same stem naming.")

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = int(round(len(pairs) * args.val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    print(f"[OK] project_root : {project_root}")
    print(f"[OK] raw_images   : {raw_images} ({len(imgs)} files)")
    print(f"[OK] matched pairs: {len(pairs)}")
    print(f"[SPLIT] train={len(train_pairs)} val={len(val_pairs)}")

    def process_one(im_path: Path, mask_path: Path, out_img_dir: Path, out_lab_dir: Path):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        bw = binarize_mask(mask, thr=int(args.mask_thr))
        lines = contours_to_yolo_seg_lines(
            bw_mask=bw,
            class_id=int(args.class_id),
            min_area_px=int(args.min_area_px),
            approx_eps_ratio=float(args.approx_eps_ratio),
        )

        (out_lab_dir / f"{im_path.stem}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "",
            encoding="utf-8",
        )

        if args.copy_images:
            shutil.copy2(im_path, out_img_dir / im_path.name)

    for im, m in tqdm(train_pairs, desc="train"):
        process_one(im, m, img_train, lab_train)

    for im, m in tqdm(val_pairs, desc="val"):
        process_one(im, m, img_val, lab_val)

    write_dataset_yaml(dataset_dir, [args.class_name])

    print("[DONE] YOLO-seg dataset ready:")
    print(f" - {dataset_dir / 'dataset.yaml'}")
    print(f" - {yolo_root}")


if __name__ == "__main__":
    main()
