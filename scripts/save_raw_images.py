# scripts/save_raw_images.py
import cv2
import argparse
import time
import re
from pathlib import Path
import sys

# ---- paths ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "images"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def next_index(out_dir: Path, prefix: str, ext: str):
    """prefix_000001.ext 기준으로 가장 큰 인덱스 + 1"""
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
    """
    backend:
      - auto: OpenCV default
      - v4l2: Linux V4L2
      - avfoundation: macOS AVFoundation
    """
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


def save_image(path: Path, img_bgr, ext: str, jpeg_quality: int):
    if ext == "jpg":
        return cv2.imwrite(
            str(path),
            img_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
    return cv2.imwrite(str(path), img_bgr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--backend", type=str, default="auto", help="auto|v4l2|avfoundation")
    ap.add_argument("--mjpg", action="store_true", help="force MJPG (useful on Ubuntu/V4L2)")
    ap.add_argument("--width", type=int, default=1280, help="capture width (0=auto)")
    ap.add_argument("--height", type=int, default=720, help="capture height (0=auto)")
    ap.add_argument("--fps", type=float, default=30, help="capture fps (0=auto)")

    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR), help="output directory")
    ap.add_argument("--prefix", type=str, default="raw", help="filename prefix")
    ap.add_argument("--ext", type=str, default="jpg", choices=["jpg", "png"], help="image extension")
    ap.add_argument("--jpeg_quality", type=int, default=95, help="jpg quality 0~100")
    ap.add_argument("--preview_scale", type=float, default=1.0,
                    help="preview scale (1.0=original). Only affects display, NOT saved image.")
    args = ap.parse_args()

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

    print(f"[OK] saving to: {out_dir.resolve()}")
    print("Keys:")
    print("  s : save current RAW frame (no warp, no resize)")
    print("  q or ESC : quit\n")

    last_save = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERR] Frame read failed.")
            break

        # preview only (doesn't affect saved image)
        vis = frame
        if args.preview_scale and abs(args.preview_scale - 1.0) > 1e-6:
            new_w = int(frame.shape[1] * args.preview_scale)
            new_h = int(frame.shape[0] * args.preview_scale)
            if new_w >= 2 and new_h >= 2:
                vis = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.putText(
            vis,
            f"RAW | next: {args.prefix}_{idx:06d}.{args.ext} | press 's' to save",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.imshow("Raw Preview", vis)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), 27]:
            break

        if key == ord("s"):
            now = time.time()
            if now - last_save < 0.15:  # double-press 방지
                continue
            last_save = now

            fname = f"{args.prefix}_{idx:06d}.{args.ext}"
            fpath = out_dir / fname
            ok = save_image(fpath, frame, args.ext, args.jpeg_quality)
            if ok:
                print(f"[SAVE] {fpath}")
                idx += 1
            else:
                print(f"[ERR] Failed to save: {fpath}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
