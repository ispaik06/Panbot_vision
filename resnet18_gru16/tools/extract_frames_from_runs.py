# resnet18_gru16/tools/extract_frames_from_runs.py

"""
Extract frames from mp4 videos inside:
  <root>/data_raw/run_XXXX/*.mp4

Features:
- Auto-detect .mp4 file inside each run folder (e.g., video_20260127_225030.mp4).
- Preserve original resolution (e.g., 3840x2160) WITHOUT resizing.
- Resample to 30 fps output (time-based sampling) and save as images:
    frames/img_000001.jpg ...
- Run range selection: --run_start / --run_end (inclusive)
- Overwrite control: --overwrite
"""

import argparse
import math
from pathlib import Path
from typing import List

import cv2


def find_mp4_in_run(run_dir: Path) -> Path:
    mp4s = sorted(run_dir.glob("*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"No .mp4 found in: {run_dir}")
    if len(mp4s) > 1:
        print(f"[WARN] Multiple mp4 found in {run_dir}. Using: {mp4s[0].name}")
    return mp4s[0]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_jpg(path: Path, img_bgr, quality: int = 95) -> bool:
    ensure_dir(path.parent)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    return bool(cv2.imwrite(str(path), img_bgr, params))


def extract_30fps_frames(
    video_path: Path,
    out_frames_dir: Path,
    overwrite: bool,
    jpg_quality: int,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Video info
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or math.isnan(src_fps) or src_fps <= 0:
        src_fps = 30.0  # fallback

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / src_fps if total_frames > 0 else None

    print(f"[VIDEO] {video_path.name}")
    print(f"  - src: {src_w}x{src_h}, fps={src_fps:.3f}, frames={total_frames}")
    if duration_sec is not None:
        print(f"  - duration: ~{duration_sec:.2f}s")
    print(f"  - out: {out_frames_dir} (target=30fps)")

    target_fps = 30.0
    next_out_t = 0.0
    out_idx = 1
    eps = 1e-7

    frame_i = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1

        if frame is None:
            continue

        # Ensure no resize happened
        if frame.shape[1] != src_w or frame.shape[0] != src_h:
            print(f"[WARN] Frame size changed: got {frame.shape[1]}x{frame.shape[0]} (expected {src_w}x{src_h})")

        t = (frame_i - 1) / src_fps

        while t + eps >= next_out_t:
            out_path = out_frames_dir / f"img_{out_idx:06d}.jpg"

            if (not overwrite) and out_path.exists():
                pass
            else:
                ok2 = save_jpg(out_path, frame, quality=jpg_quality)
                if not ok2:
                    raise RuntimeError(f"Failed to write image: {out_path}")
                saved += 1

            out_idx += 1
            next_out_t = (out_idx - 1) / target_fps

            if duration_sec is not None and next_out_t > duration_sec + 1.0:
                break

        if out_idx % 300 == 0:
            print(f"  - progress: out_idx={out_idx:06d}, saved={saved}")

    cap.release()
    print(f"[DONE] saved={saved}, last_out_idx={out_idx-1:06d}")


def iter_run_dirs(data_raw: Path, run_start: int, run_end: int) -> List[Path]:
    runs = []
    for k in range(run_start, run_end + 1):
        run_dir = data_raw / f"run_{k:04d}"
        if run_dir.exists() and run_dir.is_dir():
            runs.append(run_dir)
        else:
            print(f"[SKIP] missing: {run_dir}")
    return runs


def main():
    ap = argparse.ArgumentParser(
        description="Extract 30fps frames from run_XXXX mp4 videos (no warp)."
    )
    ap.add_argument("--root", type=str, default=None,
                    help="Project root for this CNN folder (default: auto-detect as script/..).")
    ap.add_argument("--data_raw", type=str, default="data_raw",
                    help="Relative path to data_raw directory under --root (default: data_raw).")
    ap.add_argument("--run_start", type=int, required=True, help="Start run number (e.g., 1 for run_0001).")
    ap.add_argument("--run_end", type=int, required=True, help="End run number (inclusive).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing frames if present.")
    ap.add_argument("--jpg_quality", type=int, default=95, help="JPEG quality (1-100). Default=95")

    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    default_root = script_path.parent.parent  # tools/.. = <root>
    root = Path(args.root).resolve() if args.root else default_root

    data_raw = (root / args.data_raw).resolve()
    if not data_raw.exists():
        raise FileNotFoundError(f"data_raw not found: {data_raw}")

    run_dirs = iter_run_dirs(data_raw, args.run_start, args.run_end)
    if not run_dirs:
        print("[INFO] No run directories found in the given range.")
        return

    print(f"[ROOT] {root}")
    print(f"[DATA] {data_raw}")
    print(f"[RANGE] run_{args.run_start:04d} ~ run_{args.run_end:04d}")
    print("[MODE] no-warp (original frames only)")

    for run_dir in run_dirs:
        try:
            mp4_path = find_mp4_in_run(run_dir)
        except Exception as e:
            print(f"[SKIP] {run_dir.name}: {e}")
            continue

        out_frames_dir = run_dir / "frames"
        ensure_dir(out_frames_dir)

        try:
            extract_30fps_frames(
                video_path=mp4_path,
                out_frames_dir=out_frames_dir,
                overwrite=args.overwrite,
                jpg_quality=args.jpg_quality,
            )
        except Exception as e:
            print(f"[ERROR] {run_dir.name}: {e}")
            continue


if __name__ == "__main__":
    main()
