# resnet18_gru16/tools/warp_runs_batch.py

"""
Batch warp images for multiple runs.

- Input per run:
    <root>/data_raw/run_XXXX/<in_subdir>/*.jpg|png|...
- Output per run:
    <root>/data_raw/run_XXXX/<out_subdir>/<same_stem + same_extension>

- corners.json format:
  {
    "points": [
      {"x": ..., "y": ...},
      {"x": ..., "y": ...},
      {"x": ..., "y": ...},
      {"x": ..., "y": ...}
    ]
  }

Defaults:
- Keep original extension
- If output is JPG/JPEG, save with quality=100 (minimize compression loss)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # .../resnet18_gru16


# -----------------------
# Warp utilities
# -----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_corners(corners_path: Path) -> Optional[np.ndarray]:
    if not corners_path.exists():
        print(f"[ERROR] corners.json not found: {corners_path.resolve()}")
        return None
    try:
        data = json.loads(corners_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR] failed to parse corners.json: {e}")
        return None

    pts = data.get("points", None)
    if not pts or len(pts) != 4:
        print("[ERROR] invalid corners.json format. Expect points x4 with x/y.")
        return None

    try:
        corners = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] invalid point format in corners.json: {e}")
        return None

    return corners


def compute_warp(frame_bgr, corners: np.ndarray, out_w=None, out_h=None):
    """
    corners: np.float32 (4,2) in TL,TR,BR,BL order (or any consistent order matching your corners.json).
    """
    tl, tr, br, bl = corners

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    if maxW < 50 or maxH < 50:
        return None, None, (maxW, maxH)

    if out_w is not None:
        maxW = int(out_w)
    if out_h is not None:
        maxH = int(out_h)

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame_bgr, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
    return warped, M, (maxW, maxH)


def save_image_keep_ext(path: Path, img_bgr) -> bool:
    """
    Save using original extension.
    - For JPG/JPEG: quality=100 to minimize compression loss
    - For others (PNG, etc.): default encoder settings
    """
    ensure_dir(path.parent)
    ext = path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        return bool(cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100]))
    return bool(cv2.imwrite(str(path), img_bgr))


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]


def list_images(images_dir: Path, limit: int = 0) -> List[Path]:
    files = sorted([p for p in images_dir.iterdir() if p.is_file() and is_image_file(p)])
    if limit and limit > 0:
        files = files[:limit]
    return files


# -----------------------
# Batch runner
# -----------------------
def run_one(
    run_dir: Path,
    in_subdir: str,
    out_subdir: str,
    corners: np.ndarray,
    warp_w: Optional[int],
    warp_h: Optional[int],
    preview: bool,
    preview_scale: float,
    limit: int,
    overwrite: bool,
):
    images_dir = run_dir / in_subdir
    out_dir = run_dir / out_subdir

    if not images_dir.exists():
        print(f"[SKIP] {run_dir.name}: missing input dir: {images_dir}")
        return

    ensure_dir(out_dir)

    files = list_images(images_dir, limit=limit)
    if not files:
        print(f"[SKIP] {run_dir.name}: no images in {images_dir}")
        return

    print(f"\n[RUN] {run_dir.name}")
    print(f"  in : {images_dir} (count={len(files)})")
    print(f"  out: {out_dir}")

    ok_count = 0
    fail_count = 0
    skip_count = 0

    for idx, img_path in enumerate(files, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] read fail: {img_path.name}")
            fail_count += 1
            continue

        warped, _, (wg, hg) = compute_warp(img, corners, out_w=warp_w, out_h=warp_h)
        if warped is None:
            print(f"  [WARN] warp fail: {img_path.name} (guess={wg}x{hg})")
            fail_count += 1
            continue

        out_path = out_dir / (img_path.stem + img_path.suffix)

        if (not overwrite) and out_path.exists():
            skip_count += 1
            continue

        if not save_image_keep_ext(out_path, warped):
            print(f"  [WARN] save fail: {out_path.name}")
            fail_count += 1
            continue

        ok_count += 1
        if ok_count % 50 == 0:
            print(f"  [SAVE] {ok_count} done... last={out_path.name}")

        if preview:
            vis = warped
            if preview_scale and abs(preview_scale - 1.0) > 1e-6:
                new_w = max(2, int(vis.shape[1] * preview_scale))
                new_h = max(2, int(vis.shape[0] * preview_scale))
                vis = cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_AREA)

            cv2.putText(
                vis,
                f"{run_dir.name}  {idx}/{len(files)}  saved={ok_count}  skipped={skip_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Warp Preview", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                print("[STOP] preview stopped by user")
                cv2.destroyAllWindows()
                return

    if preview:
        cv2.destroyAllWindows()

    print(f"[DONE] {run_dir.name}: saved={ok_count}, skipped={skip_count}, fail={fail_count}, out_dir={out_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Batch warp frames for run_XXXX folders (keep ext; JPG saved with quality=100)."
    )
    ap.add_argument("--root", type=str, default=None,
                    help="Project root (default: auto-detect as tools/..).")
    ap.add_argument("--data_raw", type=str, default="data_raw",
                    help="Relative path to data_raw under root (default: data_raw).")

    ap.add_argument("--run_start", type=int, required=True)
    ap.add_argument("--run_end", type=int, required=True)

    ap.add_argument("--in_subdir", type=str, default="frames",
                    help="Input subdir inside each run (default: frames).")
    ap.add_argument("--out_subdir", type=str, default="warped",
                    help="Output subdir inside each run (default: warped).")

    ap.add_argument("--corners", type=str, required=True,
                    help="Path to corners.json")

    ap.add_argument("--warp_w", type=int, default=0, help="Warp output width (0=auto).")
    ap.add_argument("--warp_h", type=int, default=0, help="Warp output height (0=auto).")

    ap.add_argument("--preview", action="store_true", help="Show preview while processing.")
    ap.add_argument("--preview_scale", type=float, default=1.0, help="Preview scale (display only).")
    ap.add_argument("--limit", type=int, default=0, help="Process first N images per run (0=no limit).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output images.")

    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    default_root = script_path.parent.parent  # tools/.. = <root>
    root = Path(args.root).resolve() if args.root else default_root
    data_raw = (root / args.data_raw).resolve()

    if not data_raw.exists():
        print(f"[ERROR] data_raw not found: {data_raw}")
        sys.exit(1)

    corners_path = Path(args.corners).expanduser().resolve()
    corners = load_corners(corners_path)
    if corners is None:
        sys.exit(1)

    warp_w = args.warp_w if args.warp_w > 0 else None
    warp_h = args.warp_h if args.warp_h > 0 else None

    print(f"[ROOT] {root}")
    print(f"[DATA] {data_raw}")
    print(f"[RANGE] run_{args.run_start:04d} ~ run_{args.run_end:04d}")
    print(f"[CORNERS] {corners_path}")
    print(f"[IN] {args.in_subdir}   [OUT] {args.out_subdir}")
    print(f"[WARP_SIZE] {warp_w if warp_w else 'auto'} x {warp_h if warp_h else 'auto'}")
    print(f"[SAVE] keep extension; if JPG/JPEG -> quality=100")
    print(f"[OVERWRITE] {args.overwrite}")
    print(f"[PREVIEW] {args.preview} (scale={args.preview_scale})  [LIMIT] {args.limit}")

    for k in range(args.run_start, args.run_end + 1):
        run_dir = data_raw / f"run_{k:04d}"
        if not run_dir.exists():
            print(f"[SKIP] missing: {run_dir}")
            continue

        run_one(
            run_dir=run_dir,
            in_subdir=args.in_subdir,
            out_subdir=args.out_subdir,
            corners=corners,
            warp_w=warp_w,
            warp_h=warp_h,
            preview=args.preview,
            preview_scale=args.preview_scale,
            limit=args.limit,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
