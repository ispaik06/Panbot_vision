import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def resolve_project_root() -> Path:
    # .../PANBOT_VISION/yolov8/scripts/predict_batter_area.py -> parents[2] = PANBOT_VISION
    return Path(__file__).resolve().parents[2]


def load_corners(corners_path: Path):
    """
    corners.json format:
    {
      "points": [{"x":..,"y":..}, ... 4 points ...]
    }
    expected order: TL, TR, BR, BL
    """
    if not corners_path.exists():
        return None
    data = json.loads(corners_path.read_text(encoding="utf-8"))
    pts = data.get("points", None)
    if not pts or len(pts) != 4:
        return None
    return np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)


def compute_warp_size_from_corners(corners: np.ndarray):
    tl, tr, br, bl = corners

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    out_w = int(round(max(width_a, width_b)))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    out_h = int(round(max(height_a, height_b)))

    out_w = max(out_w, 50)
    out_h = max(out_h, 50)
    return out_w, out_h


def warp_topview(frame_bgr: np.ndarray, corners: np.ndarray, out_w: int, out_h: int):
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame_bgr, M, (out_w, out_h))
    return warped


def pick_largest_mask(masks_bool: np.ndarray):
    """
    masks_bool: (N,H,W) bool
    returns: (H,W) bool
    """
    areas = masks_bool.reshape(masks_bool.shape[0], -1).sum(axis=1)
    idx = int(np.argmax(areas))
    return masks_bool[idx]


def overlay_mask(frame_bgr: np.ndarray, mask_bool: np.ndarray, alpha: float = 0.45):
    h, w = frame_bgr.shape[:2]

    # mask가 frame이랑 크기가 다르면 mask만 frame 크기로 맞춘다
    if mask_bool.shape[0] != h or mask_bool.shape[1] != w:
        mask_u8 = (mask_bool.astype(np.uint8) * 255)
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_u8 > 0

    overlay = frame_bgr.copy()
    overlay[mask_bool] = (0, 200, 0)
    out = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
    return out



def backend_to_cv2(backend: str) -> int:
    b = backend.lower().strip()
    if b in ("", "auto", "any"):
        return 0
    if b in ("v4l2", "cap_v4l2"):
        return cv2.CAP_V4L2
    if b in ("gstreamer", "gst", "cap_gstreamer"):
        return cv2.CAP_GSTREAMER
    if b in ("ffmpeg", "cap_ffmpeg"):
        return cv2.CAP_FFMPEG
    if b in ("msmf", "cap_msmf"):
        return cv2.CAP_MSMF
    if b in ("dshow", "cap_dshow"):
        return cv2.CAP_DSHOW
    raise ValueError(f"Unknown backend: {backend}")


def open_camera(cam_index: int, backend: str, mjpg: bool, width: int, height: int, fps: int):
    api = backend_to_cv2(backend)
    cap = cv2.VideoCapture(cam_index, api)

    if not cap.isOpened():
        return cap

    # Apply settings (best-effort; some cams/drivers may ignore)
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(fps))

    # Read back actual
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    print(f"[CAM] index={cam_index} backend={backend} mjpg={mjpg}")
    print(f"[CAM] requested {width}x{height}@{fps} -> actual {actual_w}x{actual_h}@{actual_fps:.2f} fourcc={fourcc}")
    return cap


def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    return cap


def resize_for_preview(img: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0 or abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()

    # model
    ap.add_argument("--model", type=str, required=True, help="path to best.pt (YOLOv8-seg)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)

    # input source: camera OR video
    ap.add_argument("--cam", type=int, default=-1, help="webcam index, e.g. 0, 4, 8 ...")
    ap.add_argument("--video", type=str, default="", help="video file path (if set, use this instead of --cam)")

    # camera options
    ap.add_argument("--backend", type=str, default="auto", help="auto|v4l2|gstreamer|ffmpeg|msmf|dshow")
    ap.add_argument("--mjpg", action="store_true", help="set FOURCC=MJPG")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--fps", type=int, default=0)
    ap.add_argument("--preview_scale", type=float, default=1.0, help="only affects preview window size")

    # warp options
    ap.add_argument("--use_warp", action="store_true",
                    help="If set, only warped ROI is used for inference/area (pan ROI only).")
    ap.add_argument("--corners", type=str, default="calibration/corners.json",
                    help="corners.json path (relative to project root if not absolute)")
    ap.add_argument("--warp_w", type=int, default=0, help="0 => auto from corners")
    ap.add_argument("--warp_h", type=int, default=0, help="0 => auto from corners")

    # area thresholding
    ap.add_argument("--area_thr_ratio", type=float, default=0.12,
                    help="trigger when (mask_area / frame_area) >= this")
    ap.add_argument("--hold_frames", type=int, default=5,
                help="trigger becomes True only if ratio>=thr for this many consecutive frames")

    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save_video", type=str, default="", help="optional output video path (mp4)")
    ap.add_argument("--print", action="store_true", help="print ratio every frame (can be noisy)")

    args = ap.parse_args()

    project_root = resolve_project_root()

    # ---- load warp config if needed ----
    corners = None
    warp_out_w = None
    warp_out_h = None

    if args.use_warp:
        corners_path = Path(args.corners)
        if not corners_path.is_absolute():
            corners_path = (project_root / corners_path).resolve()

        corners = load_corners(corners_path)
        if corners is None:
            raise FileNotFoundError(f"corners.json invalid/not found: {corners_path}")

        if args.warp_w > 0 and args.warp_h > 0:
            warp_out_w, warp_out_h = int(args.warp_w), int(args.warp_h)
            auto = False
        else:
            warp_out_w, warp_out_h = compute_warp_size_from_corners(corners)
            auto = True

        print(f"[OK] warp ON (pan ROI only): corners={corners_path}")
        print(f"[OK] warp size: {warp_out_w}x{warp_out_h} (auto={auto})")
    else:
        print("[OK] warp OFF (full frame inference)")

    # ---- load model ----
    model = YOLO(args.model)

    # ---- open input ----
    cap = None
    if args.video:
        src_path = Path(args.video)
        if not src_path.is_absolute():
            src_path = (project_root / src_path).resolve()
        cap = open_video(src_path)
        src_desc = f"video={src_path}"
    else:
        if args.cam < 0:
            raise ValueError("Provide --cam <index> OR --video <path>")
        cap = open_camera(args.cam, args.backend, args.mjpg, args.width, args.height, args.fps)
        src_desc = f"cam={args.cam}"

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {src_desc}")

    # ---- optional writer ----
    writer = None
    if args.save_video:
        out_path = Path(args.save_video)
        if not out_path.is_absolute():
            out_path = (project_root / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-3:
            fps = 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if args.use_warp:
            out_w, out_h = int(warp_out_w), int(warp_out_h)
        else:
            out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (out_w, out_h))
        print(f"[OK] saving video -> {out_path} ({out_w}x{out_h} @ {fps:.2f}fps)")

    print("Press 'q' to quit.")
    hit_count = 0
    stable_trigger = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ 팬 ROI만 보도록: 워프를 쓰면 프레임 자체를 워프 결과로 교체
        if args.use_warp:
            frame = warp_topview(frame, corners, warp_out_w, warp_out_h)

        H, W = frame.shape[:2]
        frame_area = float(H * W)

        # inference
        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        r = results[0]

        mask_area = 0.0
        ratio = 0.0
        triggered = False
        vis = frame

        if r.masks is not None and r.masks.data is not None and len(r.masks.data) > 0:
            masks = r.masks.data.detach().cpu().numpy()  # (N,h,w) float 0..1
            masks_bool = masks > 0.5
            largest = pick_largest_mask(masks_bool)  # (h,w) but may differ from frame size

            # ✅ largest를 frame 크기(H,W)로 맞추기
            if largest.shape[0] != H or largest.shape[1] != W:
                largest_u8 = (largest.astype(np.uint8) * 255)
                largest_u8 = cv2.resize(largest_u8, (W, H), interpolation=cv2.INTER_NEAREST)
                largest = largest_u8 > 0

            mask_area = float(largest.sum())
            ratio = mask_area / frame_area
            if ratio >= float(args.area_thr_ratio):
                hit_count += 1
            else:
                hit_count = 0

            stable_trigger = hit_count >= int(args.hold_frames)
            # stable_trigger = stable_trigger or (hit_count >= int(args.hold_frames))
            triggered = stable_trigger


            vis = overlay_mask(frame, largest)


        text = f"area={mask_area:.0f}px  ratio={ratio:.3f}  thr={args.area_thr_ratio:.3f}  TRIGGER={triggered}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        if writer is not None:
            writer.write(vis)

        if args.show:
            preview = resize_for_preview(vis, float(args.preview_scale))
            cv2.imshow("batter_area", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break

        if args.print:
            print(text)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[DONE]")


if __name__ == "__main__":
    main()
