# scripts/capture_warped_images.py
import cv2
import numpy as np
import json
from pathlib import Path
import argparse
import sys
import time
import re

# ---- paths ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CORNERS = PROJECT_ROOT / "calibration" / "corners.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "images"


def load_corners(corners_path: Path):
    if not corners_path.exists():
        return None
    try:
        data = json.loads(corners_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    pts = data.get("points", None)
    if not pts or len(pts) != 4:
        return None

    corners = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)  # TL,TR,BR,BL
    return corners


def compute_warp(frame, corners, out_w=None, out_h=None):
    tl, tr, br, bl = corners

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    if maxW < 50 or maxH < 50:
        return None, None, (maxW, maxH)

    # out_w/out_h가 None이면 자동(maxW/maxH) 사용
    if out_w is not None:
        maxW = int(out_w)
    if out_h is not None:
        maxH = int(out_h)

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, M, (maxW, maxH))
    return warped, M, (maxW, maxH)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def next_index(out_dir: Path, prefix: str, ext: str):
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    best = 0
    for p in out_dir.glob(f"{prefix}_*.{ext}"):
        m = pat.match(p.stem)
        if not m:
            continue
        try:
            best = max(best, int(m.group(1)))
        except Exception:
            pass
    return best + 1


def open_capture(cam_index: int, backend: str):
    b = (backend or "auto").lower()
    if b == "auto":
        return cv2.VideoCapture(cam_index)
    if b == "v4l2":
        return cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if b == "avfoundation":
        return cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)
    raise ValueError(f"Unknown backend: {backend} (use auto|v4l2|avfoundation)")


def apply_capture_settings(cap, width, height, fps, mjpg: bool):
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, float(fps))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    try:
        backend_name = cap.getBackendName()
    except Exception:
        backend_name = "(unknown)"

    print(f"[Camera] backend={backend_name}")
    print(f"[Camera] requested: {width or 'auto'}x{height or 'auto'} @ {fps or 'auto'}fps, mjpg={mjpg}")
    print(f"[Camera] actual:    {actual_w}x{actual_h} @ {actual_fps:.2f}fps")


def resize_for_preview(img_bgr, scale: float):
    """Display-only resize. Does NOT affect saving."""
    if img_bgr is None:
        return None
    if scale is None or abs(scale - 1.0) < 1e-6:
        return img_bgr
    if scale <= 0:
        return img_bgr
    new_w = int(img_bgr.shape[1] * scale)
    new_h = int(img_bgr.shape[0] * scale)
    if new_w < 2 or new_h < 2:
        return img_bgr
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index")

    ap.add_argument("--corners", type=str, default=str(DEFAULT_CORNERS), help="path to corners.json")
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR), help="output directory")

    # ✅ 0=auto (옵션 안 넣으면 자동 워프 크기)
    ap.add_argument("--warp_w", type=int, default=0, help="warp output width (0=auto)")
    ap.add_argument("--warp_h", type=int, default=0, help="warp output height (0=auto)")

    ap.add_argument("--prefix", type=str, default="img", help="filename prefix")
    ap.add_argument("--ext", type=str, default="jpg", choices=["jpg", "png"], help="image extension")
    ap.add_argument("--jpeg_quality", type=int, default=95, help="jpg quality 0~100")

    ap.add_argument("--backend", type=str, default="auto", help="auto|v4l2|avfoundation")
    ap.add_argument("--mjpg", action="store_true", help="force MJPG (useful on Ubuntu/V4L2)")
    ap.add_argument("--width", type=int, default=1280, help="capture width (0=auto)")
    ap.add_argument("--height", type=int, default=720, help="capture height (0=auto)")
    ap.add_argument("--fps", type=float, default=30, help="capture fps (0=auto)")

    ap.add_argument("--show_raw", action="store_true", help="also show raw preview window")

    # ✅ preview scale (display only)
    ap.add_argument(
        "--preview_scale",
        type=float,
        default=1.0,
        help="preview scale (display only). Example: 0.5, 1.0, 1.5",
    )

    args = ap.parse_args()

    corners_path = Path(args.corners)
    corners = load_corners(corners_path)
    if corners is None:
        print(f"[ERROR] corners.json not found or invalid: {corners_path.resolve()}")
        print("[EXIT] Please run calibration and save corners.json first.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    idx = next_index(out_dir, args.prefix, args.ext)

    cap = open_capture(args.cam, args.backend)
    if not cap.isOpened():
        print(f"[ERROR] camera open failed. Try --cam 1 or 2 (current: {args.cam})")
        sys.exit(1)

    req_w = args.width if args.width > 0 else None
    req_h = args.height if args.height > 0 else None
    req_fps = args.fps if args.fps and args.fps > 0 else None
    apply_capture_settings(cap, req_w, req_h, req_fps, args.mjpg)

    out_w = args.warp_w if args.warp_w > 0 else None
    out_h = args.warp_h if args.warp_h > 0 else None

    warp_desc = "AUTO" if (out_w is None and out_h is None) else f"{out_w}x{out_h}"
    print(f"[OK] corners loaded: {corners_path.resolve()}")
    print(f"[OK] saving to: {out_dir.resolve()}")
    print(f"[OK] warp size: {warp_desc}")
    if abs(args.preview_scale - 1.0) > 1e-6:
        print(f"[OK] preview_scale: {args.preview_scale} (display only)")
    print("Keys:")
    print("  s : save current warped frame (saved image is NOT scaled by preview_scale)")
    print("  q or ESC : quit\n")

    last_save_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERR] Frame read failed.")
            break

        warped, _, (w, h) = compute_warp(frame, corners, out_w=out_w, out_h=out_h)

        if args.show_raw:
            raw_vis = frame.copy()
            cv2.putText(raw_vis, "RAW", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            raw_disp = resize_for_preview(raw_vis, args.preview_scale)
            cv2.imshow("Raw", raw_disp)

        if warped is None:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(blank, f"Warp failed ({w}x{h})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            blank_disp = resize_for_preview(blank, args.preview_scale)
            cv2.imshow("Warped (Top-View)", blank_disp)
        else:
            vis = warped.copy()
            cv2.putText(
                vis,
                f"WARP {vis.shape[1]}x{vis.shape[0]} | next: {args.prefix}_{idx:06d}.{args.ext}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            vis_disp = resize_for_preview(vis, args.preview_scale)
            cv2.imshow("Warped (Top-View)", vis_disp)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), 27]:
            break

        if key == ord("s"):
            now = time.time()
            if now - last_save_time < 0.15:
                continue
            last_save_time = now

            if warped is None:
                print("[WARN] Warp is None. Not saved.")
                continue

            fname = f"{args.prefix}_{idx:06d}.{args.ext}"
            fpath = out_dir / fname

            if args.ext == "jpg":
                ok = cv2.imwrite(
                    str(fpath),
                    warped,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
                )
            else:
                ok = cv2.imwrite(str(fpath), warped)

            if ok:
                print(f"[SAVE] {fpath}")
                idx += 1
            else:
                print(f"[ERR] Failed to save: {fpath}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
