# tools/video_to_frames.py
import cv2
import numpy as np
import json
import argparse
import sys
import time
import re
from pathlib import Path
from typing import Optional

# ---- paths ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "images"
DEFAULT_VIDEOS_DIR = PROJECT_ROOT / "videos"

# batch default
DEFAULT_BATCH_OUT_ROOT = PROJECT_ROOT / "raw_datasets" / "raw_images"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def next_index(out_dir: Path, prefix: str, ext: str):
    """prefix_000001.ext 형태 기준으로 가장 큰 인덱스 + 1 반환"""
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
    return np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)  # TL,TR,BR,BL


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


def save_image(path: Path, img_bgr, ext: str, jpeg_quality: int):
    if ext == "jpg":
        return cv2.imwrite(
            str(path),
            img_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
    return cv2.imwrite(str(path), img_bgr)


def open_capture_video(video_path: Path):
    return cv2.VideoCapture(str(video_path))


def open_capture_cam(cam_index: int, backend: str):
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


def record_to_video(cap, out_path: Path, duration_s: float, fps: float, width: int, height: int):
    ensure_dir(out_path.parent)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        print("[ERROR] VideoWriter open failed. Try different codec/container.")
        return False

    print(f"[REC] saving video -> {out_path.resolve()}  (duration={duration_s}s)")
    t0 = time.time()
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        frames += 1
        if time.time() - t0 >= duration_s:
            break

    writer.release()
    print(f"[REC] done. frames={frames}")
    return True


# -----------------------------
# Batch helpers
# -----------------------------
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
SOURCE_TAG = "__source_video.txt"


def parse_datetime_from_name(name: str):
    m = re.search(r"(20\d{6})(?:[_-]?(\d{6}))?", name)
    if not m:
        return None
    return (m.group(1), m.group(2))  # (date, time or None)


def collect_videos_for_date(videos_dir: Path, yyyymmdd: str):
    vids = []
    for p in videos_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        if yyyymmdd not in p.name:
            continue
        vids.append(p)

    def sort_key(p: Path):
        dt = parse_datetime_from_name(p.name)
        if dt and dt[0] == yyyymmdd and dt[1] is not None:
            return (0, dt[1], p.name)
        if dt and dt[0] == yyyymmdd:
            return (1, p.name)
        return (2, p.stat().st_mtime, p.name)

    vids.sort(key=sort_key)
    return vids


def list_existing_numeric_folders(date_root: Path):
    nums = []
    if not date_root.exists():
        return nums
    for p in date_root.iterdir():
        if p.is_dir() and p.name.isdigit():
            nums.append(int(p.name))
    nums.sort()
    return nums


def read_source_tag(folder: Path) -> Optional[str]:
    tag = folder / SOURCE_TAG
    if not tag.exists():
        return None
    try:
        return tag.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def build_already_done_map(date_root: Path):
    done = set()
    for n in list_existing_numeric_folders(date_root):
        folder = date_root / str(n)
        src = read_source_tag(folder)
        if src:
            done.add(src)
    return done


def write_source_tag(folder: Path, video_path: Path):
    ensure_dir(folder)
    (folder / SOURCE_TAG).write_text(str(video_path.resolve()), encoding="utf-8")


def find_next_free_folder_index(date_root: Path):
    nums = list_existing_numeric_folders(date_root)
    return (max(nums) + 1) if nums else 1


# -----------------------------
# Core extraction (single video)
# -----------------------------
def extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    prefix: str,
    ext: str,
    jpeg_quality: int,
    corners,
    no_warp: bool,
    warp_w: int,
    warp_h: int,
    out_fps: float,
    every_n: int,
    start_s: float,
    end_s: float,
    max_frames: int,
    preview: bool,
):
    ensure_dir(out_dir)
    idx = next_index(out_dir, prefix, ext)

    out_w = warp_w if (warp_w and warp_w > 0) else None
    out_h = warp_h if (warp_h and warp_h > 0) else None

    cap = open_capture_video(video_path)
    if not cap.isOpened():
        print(f"[ERROR] failed to open video: {video_path.resolve()}")
        return 0

    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    if vid_fps <= 1e-3:
        vid_fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_est = frame_count / vid_fps if frame_count > 0 else 0.0

    warp_on = (corners is not None) and (not no_warp)

    print(f"\n[OK] video: {video_path.resolve()}")
    print(f"[VID] fps={vid_fps:.2f}, frames={frame_count}, duration≈{duration_est:.2f}s")
    print(f"[OK] out_dir: {out_dir.resolve()}  (next index={idx})")
    warp_desc = "OFF" if not warp_on else f"ON ({out_w if out_w is not None else 'auto'}x{out_h if out_h is not None else 'auto'})"
    print(f"[OK] warp: {warp_desc}")
    print(f"[OK] sampling: every_n={every_n}  out_fps={out_fps}")
    if preview:
        print("Press 'q' to stop (preview mode).")

    interval_s = (1.0 / out_fps) if (out_fps and out_fps > 0) else None
    next_t = start_s

    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)

    saved = 0
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = pos_msec / 1000.0 if pos_msec > 0 else (start_s + i / vid_fps)

        if end_s and end_s > 0 and t > end_s:
            break

        take = False
        if every_n and every_n > 0:
            if i % every_n == 0:
                take = True
        else:
            if interval_s is None:
                take = True
            else:
                if t >= next_t - 1e-6:
                    take = True
                    next_t += interval_s

        i += 1
        if not take:
            continue

        view = frame
        if warp_on:
            warped, _, (w_guess, h_guess) = compute_warp(frame, corners, out_w=out_w, out_h=out_h)
            if warped is None:
                print(f"[WARN] warp failed at t={t:.2f}s (guess={w_guess}x{h_guess}) -> skip")
                continue
            view = warped

        fname = f"{prefix}_{idx:06d}.{ext}"
        fpath = out_dir / fname
        ok = save_image(fpath, view, ext, jpeg_quality)
        if ok:
            saved += 1
            idx += 1
            if saved % 20 == 0:
                print(f"[SAVE] {saved} frames ... last={fpath.name} (t={t:.2f}s)")
        else:
            print(f"[ERR] failed to save: {fpath}")

        if preview:
            vis = view.copy()
            cv2.putText(
                vis,
                f"t={t:.2f}s  saved={saved}  next={prefix}_{idx:06d}.{ext}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Extract Preview", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break

        if max_frames and max_frames > 0 and saved >= max_frames:
            break

    cap.release()
    if preview:
        cv2.destroyAllWindows()
    print(f"[DONE] saved frames: {saved}")
    return saved


def main():
    ap = argparse.ArgumentParser()

    # --- Input source: video file OR webcam record ---
    ap.add_argument("--video", type=str, default="", help="input video path (mp4/avi/etc). If empty, you can record from webcam.")
    ap.add_argument("--cam", type=int, default=-1, help="webcam index for recording (use >=0). Example: --cam 0")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds to record from webcam (e.g. 60). If 0, no recording.")
    ap.add_argument("--record_out", type=str, default=str(DEFAULT_VIDEOS_DIR / "capture.mp4"),
                    help="output video path when recording from webcam")

    # webcam options
    ap.add_argument("--backend", type=str, default="auto", help="auto|v4l2|avfoundation")
    ap.add_argument("--mjpg", action="store_true", help="force MJPG (useful on Ubuntu/V4L2)")
    ap.add_argument("--width", type=int, default=1280, help="webcam capture width (0=auto)")
    ap.add_argument("--height", type=int, default=720, help="webcam capture height (0=auto)")
    ap.add_argument("--fps", type=float, default=30, help="webcam capture fps (0=auto)")

    # --- Warp (OPTIONAL) ---
    # corners를 "기본값 없음"으로 바꿈: 안 주면 warp 자동 OFF
    ap.add_argument("--corners", type=str, default="",
                    help="(optional) corners.json path. If empty, warp is OFF and raw frames are saved.")
    ap.add_argument("--no_warp", action="store_true", help="force warp OFF even if --corners is provided")
    ap.add_argument("--warp_w", type=int, default=640, help="warp output width (0=auto)")
    ap.add_argument("--warp_h", type=int, default=480, help="warp output height (0=auto)")

    # --- Extraction options ---
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR), help="output frames directory (single video mode)")
    ap.add_argument("--prefix", type=str, default="img", help="output filename prefix")
    ap.add_argument("--ext", type=str, default="jpg", choices=["jpg", "png"], help="image extension")
    ap.add_argument("--jpeg_quality", type=int, default=95, help="jpg quality 0~100")

    # sampling
    ap.add_argument("--out_fps", type=float, default=2.0,
                    help="extract rate in frames/sec based on video time (e.g. 2.0 => every 0.5 sec). Set 0 to disable.")
    ap.add_argument("--every_n", type=int, default=0,
                    help="extract every N frames (overrides out_fps if >0). Example: 10 => save every 10th frame.")
    ap.add_argument("--start_s", type=float, default=0.0, help="start time (sec)")
    ap.add_argument("--end_s", type=float, default=0.0, help="end time (sec). 0 means until end.")
    ap.add_argument("--max_frames", type=int, default=0, help="max number of saved frames (0=no limit)")
    ap.add_argument("--preview", action="store_true", help="show preview window while extracting")

    # --- Batch mode ---
    ap.add_argument("--batch_date", type=str, default="",
                    help="배치 모드: YYYYMMDD (예: 20260126). videos/에서 해당 날짜 동영상 모두 찾아 신규만 변환")
    ap.add_argument("--videos_dir", type=str, default=str(DEFAULT_VIDEOS_DIR),
                    help="배치 모드에서 탐색할 videos 폴더")
    ap.add_argument("--batch_out_root", type=str, default=str(DEFAULT_BATCH_OUT_ROOT),
                    help="배치 모드 출력 루트 (기본: project_root/raw_datasets/raw_images)")
    ap.add_argument("--skip_done", action="store_true",
                    help="이미 변환된 비디오는 스킵 (__source_video.txt 기반).")

    args = ap.parse_args()

    # ---- corners: provided -> load, else None (warp OFF) ----
    corners = None
    if args.corners and args.corners.strip():
        cpath = Path(args.corners).expanduser().resolve()
        corners = load_corners(cpath)
        if corners is None:
            print(f"[ERROR] corners.json not found/invalid: {cpath}")
            print("=> --corners를 빼고 실행하면 warp 없이 원본 프레임만 저장합니다.")
            sys.exit(1)

    # -----------------------------
    # Batch mode
    # -----------------------------
    if args.batch_date:
        yyyymmdd = args.batch_date.strip()
        if not re.fullmatch(r"\d{8}", yyyymmdd):
            print(f"[ERROR] --batch_date는 YYYYMMDD 형식이어야 합니다. 입력: {yyyymmdd}")
            sys.exit(1)

        videos_dir = Path(args.videos_dir)
        if not videos_dir.exists():
            print(f"[ERROR] videos_dir가 없습니다: {videos_dir.resolve()}")
            sys.exit(1)

        videos = collect_videos_for_date(videos_dir, yyyymmdd)
        if not videos:
            print(f"[ERROR] videos_dir에서 날짜 {yyyymmdd}가 포함된 동영상을 찾지 못했습니다: {videos_dir.resolve()}")
            sys.exit(1)

        date_root = Path(args.batch_out_root) / yyyymmdd
        ensure_dir(date_root)

        done_map = build_already_done_map(date_root) if args.skip_done else set()
        next_folder = find_next_free_folder_index(date_root)

        warp_on = (corners is not None) and (not args.no_warp)
        print(f"[BATCH] date={yyyymmdd}")
        print(f"[BATCH] videos_dir={videos_dir.resolve()}")
        print(f"[BATCH] out_root={date_root.resolve()}")
        print(f"[BATCH] found {len(videos)} videos")
        print(f"[BATCH] warp={'ON' if warp_on else 'OFF'} (corners={'given' if corners is not None else 'none'}, no_warp={args.no_warp})")
        if args.skip_done:
            print(f"[BATCH] skip_done=ON (tag file: {SOURCE_TAG})  already_done={len(done_map)}")
        else:
            print("[BATCH] skip_done=OFF")

        new_jobs = 0
        for vp in videos:
            vp_key = str(vp.resolve())
            if args.skip_done and vp_key in done_map:
                print(f"[SKIP] already converted: {vp.name}")
                continue

            folder_no = next_folder
            out_dir = date_root / str(folder_no)
            next_folder += 1

            # 태그 기록
            write_source_tag(out_dir, vp)

            # ✅ 폴더 번호를 파일명 prefix에 포함
            batch_prefix = f"{args.prefix}_{folder_no}"   # 기본 args.prefix="img" -> "img_1", "img_2", ...

            print(f"\n[BATCH] {vp.name} -> {out_dir}")
            saved = extract_frames_from_video(
                video_path=vp,
                out_dir=out_dir,
                prefix=batch_prefix,         # ✅ 여기!
                ext=args.ext,
                jpeg_quality=args.jpeg_quality,
                corners=corners,
                no_warp=args.no_warp,
                warp_w=args.warp_w,
                warp_h=args.warp_h,
                out_fps=args.out_fps,
                every_n=args.every_n,
                start_s=args.start_s,
                end_s=args.end_s,
                max_frames=args.max_frames,
                preview=args.preview,
            )
            if saved == 0:
                print(f"[WARN] no frames saved for {vp.name}")
            new_jobs += 1

        if new_jobs == 0:
            print("\n[BATCH] new videos 없음 (전부 이미 변환됨).")
        print("\n[BATCH DONE]")
        return

    # -----------------------------
    # Single video mode
    # -----------------------------
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    video_path = Path(args.video) if args.video else None

    if (not video_path or not video_path.exists()) and args.cam >= 0 and args.duration > 0:
        cap_cam = open_capture_cam(args.cam, args.backend)
        if not cap_cam.isOpened():
            print(f"[ERROR] camera open failed. Try --cam 1 or 2 (current: {args.cam})")
            sys.exit(1)

        req_w = args.width if args.width > 0 else None
        req_h = args.height if args.height > 0 else None
        req_fps = args.fps if args.fps and args.fps > 0 else None
        apply_capture_settings(cap_cam, req_w, req_h, req_fps, args.mjpg)

        actual_w = int(cap_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap_cam.get(cv2.CAP_PROP_FPS)
        if actual_fps <= 1e-3:
            actual_fps = float(req_fps or 30.0)

        rec_out = Path(args.record_out)
        ok = record_to_video(cap_cam, rec_out, args.duration, actual_fps, actual_w, actual_h)
        cap_cam.release()
        if not ok:
            sys.exit(1)

        video_path = rec_out

    if not video_path or not video_path.exists():
        print("[ERROR] No valid --video provided and no webcam recording done.")
        print("Example 1) Extract from existing video (raw frames):")
        print("  python tools/video_to_frames.py --video videos/pour.mp4 --out_dir images --out_fps 2")
        print("Example 2) Extract with warp (provide corners):")
        print("  python tools/video_to_frames.py --video videos/pour.mp4 --corners calibration/corners.json --out_dir images --out_fps 2")
        print("Example 3) Batch (skip converted) raw frames:")
        print("  python tools/video_to_frames.py --batch_date 20260126 --skip_done --out_fps 2")
        sys.exit(1)

    extract_frames_from_video(
        video_path=video_path,
        out_dir=out_dir,
        prefix=args.prefix,
        ext=args.ext,
        jpeg_quality=args.jpeg_quality,
        corners=corners,
        no_warp=args.no_warp,
        warp_w=args.warp_w,
        warp_h=args.warp_h,
        out_fps=args.out_fps,
        every_n=args.every_n,
        start_s=args.start_s,
        end_s=args.end_s,
        max_frames=args.max_frames,
        preview=args.preview,
    )


if __name__ == "__main__":
    main()
