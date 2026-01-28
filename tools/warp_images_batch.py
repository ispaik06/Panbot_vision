# tools/warp_images_batch.py
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import sys

# ---- paths ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CORNERS = PROJECT_ROOT / "calibration" / "corners.json"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_corners(corners_path: Path):
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

    corners = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)  # TL,TR,BR,BL
    return corners


def compute_warp(frame, corners, out_w=None, out_h=None):
    # corners: TL,TR,BR,BL (np.float32 shape (4,2))
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
    warped = cv2.warpPerspective(frame, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
    return warped, M, (maxW, maxH)


def save_image(path: Path, img_bgr, ext: str, jpeg_quality: int):
    ext = ext.lower()
    if ext in [".jpg", ".jpeg"]:
        return cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    return cv2.imwrite(str(path), img_bgr)


def is_image_file(p: Path):
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True, help="input image folder")
    ap.add_argument("--out_dir", type=str, required=True, help="output folder for warped images")
    ap.add_argument("--corners", type=str, default=str(DEFAULT_CORNERS), help="path to corners.json")

    ap.add_argument("--warp_w", type=int, default=0, help="warp output width (0=auto)")
    ap.add_argument("--warp_h", type=int, default=0, help="warp output height (0=auto)")

    ap.add_argument("--ext", type=str, default="", help="output extension: .jpg/.png (empty=keep original ext)")
    ap.add_argument("--jpeg_quality", type=int, default=95, help="jpg quality 0~100")

    ap.add_argument("--preview", action="store_true", help="show preview while processing")
    ap.add_argument("--preview_scale", type=float, default=1.0, help="preview scale only (display only)")
    ap.add_argument("--limit", type=int, default=0, help="process first N images only (0=no limit)")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    corners_path = Path(args.corners)

    if not images_dir.exists():
        print(f"[ERROR] images_dir not found: {images_dir.resolve()}")
        sys.exit(1)

    ensure_dir(out_dir)

    corners = load_corners(corners_path)
    if corners is None:
        sys.exit(1)

    out_w = args.warp_w if args.warp_w > 0 else None
    out_h = args.warp_h if args.warp_h > 0 else None

    files = sorted([p for p in images_dir.iterdir() if p.is_file() and is_image_file(p)])
    if not files:
        print(f"[ERROR] no images found in: {images_dir.resolve()}")
        sys.exit(1)

    if args.limit and args.limit > 0:
        files = files[: args.limit]

    print(f"[OK] images_dir : {images_dir.resolve()}  (count={len(files)})")
    print(f"[OK] out_dir    : {out_dir.resolve()}")
    print(f"[OK] corners    : {corners_path.resolve()}")
    warp_desc = f"{out_w if out_w else 'auto'} x {out_h if out_h else 'auto'}"
    print(f"[OK] warp size  : {warp_desc}")
    if args.preview:
        print(f"[OK] preview    : ON (scale={args.preview_scale})")
    else:
        print("[OK] preview    : OFF")

    ok_count = 0
    fail_count = 0

    for idx, img_path in enumerate(files, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] failed to read: {img_path.name}")
            fail_count += 1
            continue

        warped, _, (wg, hg) = compute_warp(img, corners, out_w=out_w, out_h=out_h)
        if warped is None:
            print(f"[WARN] warp failed: {img_path.name} (guess={wg}x{hg})")
            fail_count += 1
            continue

        out_ext = args.ext.strip().lower()
        if out_ext and not out_ext.startswith("."):
            out_ext = "." + out_ext
        if not out_ext:
            out_ext = img_path.suffix  # keep original

        out_path = out_dir / (img_path.stem + out_ext)

        if not save_image(out_path, warped, out_ext, args.jpeg_quality):
            print(f"[WARN] failed to save: {out_path.name}")
            fail_count += 1
            continue

        ok_count += 1
        if ok_count % 20 == 0:
            print(f"[SAVE] {ok_count} done... last={out_path.name}")

        if args.preview:
            vis = warped
            if args.preview_scale and abs(args.preview_scale - 1.0) > 1e-6:
                new_w = max(2, int(vis.shape[1] * args.preview_scale))
                new_h = max(2, int(vis.shape[0] * args.preview_scale))
                vis = cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_AREA)

            cv2.putText(
                vis,
                f"{idx}/{len(files)}  saved={ok_count}  {out_path.name}",
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
                break

    if args.preview:
        cv2.destroyAllWindows()

    print(f"[DONE] ok={ok_count}, fail={fail_count}, out_dir={out_dir.resolve()}")


if __name__ == "__main__":
    main()
