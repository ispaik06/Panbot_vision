# scripts/record_video.py
import cv2
import argparse
import time
from pathlib import Path
import sys
import threading

# ---- paths ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_VIDEOS_DIR = PROJECT_ROOT / "videos"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


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

    return actual_w, actual_h, actual_fps


def build_writer(out_path: Path, fps: float, width: int, height: int, codec: str):
    """
    codec:
      - mp4v (default): mp4 container
      - avc1: H.264 (환경에 따라 안될 수 있음)
      - XVID: avi에 자주 씀
    """
    c = codec.lower()
    if c == "mp4v":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif c == "avc1":
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
    elif c == "xvid":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        raise ValueError("codec must be one of: mp4v|avc1|xvid")

    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(width), int(height)))
    return writer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--backend", type=str, default="auto", help="auto|v4l2|avfoundation")
    ap.add_argument("--mjpg", action="store_true", help="force MJPG (useful on Ubuntu/V4L2)")

    ap.add_argument("--width", type=int, default=1280, help="capture width (0=auto)")
    ap.add_argument("--height", type=int, default=720, help="capture height (0=auto)")
    ap.add_argument("--fps", type=float, default=30, help="capture fps (0=auto)")

    ap.add_argument("--duration", type=float, default=60, help="record duration (seconds)")
    ap.add_argument(
        "--until_key",
        action="store_true",
        help="record until key press: (preview ON) q/ESC, (preview OFF) press Enter in terminal",
    )

    ap.add_argument("--out", type=str, default="", help="output video path. If empty, save to PROJECT_ROOT/videos/")
    ap.add_argument("--name", type=str, default="video", help="filename stem when --out is empty")
    ap.add_argument("--codec", type=str, default="mp4v", help="mp4v|avc1|xvid")

    ap.add_argument("--preview", action="store_true", help="show preview window while recording")
    ap.add_argument(
        "--preview_scale",
        type=float,
        default=0.6,
        help="preview resize scale (only display, not saved). e.g., 0.5",
    )

    args = ap.parse_args()

    ensure_dir(DEFAULT_VIDEOS_DIR)

    # output path
    if args.out:
        out_path = Path(args.out)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        ext = ".mp4" if args.codec.lower() in ["mp4v", "avc1"] else ".avi"
        out_path = DEFAULT_VIDEOS_DIR / f"{args.name}_{ts}{ext}"

    ensure_dir(out_path.parent)

    cap = open_capture(args.cam, args.backend)
    if not cap.isOpened():
        print(f"[ERROR] camera open failed. Try --cam 1 or 2 (current: {args.cam})")
        sys.exit(1)

    req_w = args.width if args.width > 0 else None
    req_h = args.height if args.height > 0 else None
    req_fps = args.fps if args.fps and args.fps > 0 else None
    actual_w, actual_h, actual_fps = apply_capture_settings(cap, req_w, req_h, req_fps, args.mjpg)

    # fps fallback
    fps = actual_fps if actual_fps and actual_fps > 1e-3 else float(req_fps or 30.0)

    writer = build_writer(out_path, fps, actual_w, actual_h, args.codec)
    if not writer.isOpened():
        print(f"[ERROR] VideoWriter open failed for {out_path}")
        print("Try:")
        print("  - change --codec mp4v / xvid")
        print("  - or change output extension/container")
        cap.release()
        sys.exit(1)

    stop_flag = {"stop": False}

    # preview OFF + until_key ON: 터미널 Enter로 종료
    def wait_for_enter():
        try:
            input("[WAIT] Press Enter to stop recording...\n")
        except Exception:
            pass
        stop_flag["stop"] = True

    if args.until_key and (not args.preview):
        t = threading.Thread(target=wait_for_enter, daemon=True)
        t.start()

    print(f"[REC] output: {out_path.resolve()}")
    if args.until_key:
        if args.preview:
            print("Keys: q/ESC to stop early (preview window)\n")
        else:
            print("Keys: press Enter in terminal to stop\n")
    else:
        print("Keys (preview mode): q/ESC to stop early\n")

    t0 = time.time()
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERR] Frame read failed.")
            break

        # ensure size match writer
        if frame.shape[1] != actual_w or frame.shape[0] != actual_h:
            frame = cv2.resize(frame, (actual_w, actual_h), interpolation=cv2.INTER_AREA)

        writer.write(frame)
        frames += 1

        elapsed = time.time() - t0

        if args.preview:
            vis = frame.copy()

            # preview만 축소(저장 영상은 원본 해상도 그대로)
            s = float(args.preview_scale)
            if s > 0 and abs(s - 1.0) > 1e-6:
                new_w = max(2, int(vis.shape[1] * s))
                new_h = max(2, int(vis.shape[0] * s))
                vis = cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if args.until_key:
                label = f"REC {elapsed:.1f}s  frames={frames}  (q/ESC to stop)"
            else:
                label = f"REC {elapsed:.1f}s / {args.duration:.1f}s  frames={frames}"

            cv2.putText(
                vis,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Recording Preview", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                print("[STOP] stopped by user")
                break

        # 종료 조건
        if args.until_key:
            if stop_flag["stop"]:
                print("[STOP] stopped by Enter")
                break
        else:
            if elapsed >= args.duration:
                break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"[DONE] saved: {out_path.resolve()}")
    print(f"[DONE] frames={frames}, elapsed={elapsed:.2f}s, fps≈{frames/max(elapsed,1e-6):.2f}")


if __name__ == "__main__":
    main()
