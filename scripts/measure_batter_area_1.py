import cv2
import numpy as np
import json
from pathlib import Path
import argparse
from collections import deque

# ---- paths ----
SCRIPT_DIR = Path(__file__).resolve().parent         # .../scripts
PROJECT_ROOT = SCRIPT_DIR.parent                     # .../pancake_vision
DEFAULT_CORNERS = PROJECT_ROOT / "calibration" / "corners.json"


def load_corners(corners_path: Path):
    if not corners_path.exists():
        return None
    data = json.loads(corners_path.read_text(encoding="utf-8"))
    pts = data.get("points", None)
    if not pts or len(pts) != 4:
        return None
    corners = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)
    return corners  # TL,TR,BR,BL


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
    warped = cv2.warpPerspective(frame, M, (maxW, maxH))
    return warped, M, (maxW, maxH)


def largest_component(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    return c, area


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

    return actual_w, actual_h


def build_view(frame, corners, out_w, out_h):
    """
    RAW -> (if corners) WARP
    returns: view, mode_string
    """
    if corners is None:
        return frame, "RAW(no corners)"

    warped, _, wh = compute_warp(frame, corners, out_w=out_w, out_h=out_h)
    if warped is not None:
        return warped, f"WARP({warped.shape[1]}x{warped.shape[0]})"
    return frame, f"WARP_FAIL(guess={wh[0]}x{wh[1]})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)

    # ✅ 기본 경로: 프로젝트폴더/calibration/corners.json
    ap.add_argument("--corners", type=str, default=str(DEFAULT_CORNERS))

    ap.add_argument("--bg", type=str, default="bg.png")

    ap.add_argument("--diff_thresh", type=int, default=25)
    ap.add_argument("--min_area", type=int, default=800)
    ap.add_argument("--stable_frames", type=int, default=5)
    ap.add_argument("--smooth", type=int, default=7)
    ap.add_argument("--warp_w", type=int, default=0)
    ap.add_argument("--warp_h", type=int, default=0)

    # ✅ 캡처 고정(중요)
    ap.add_argument("--width", type=int, default=1280, help="capture width (0=auto)")
    ap.add_argument("--height", type=int, default=720, help="capture height (0=auto)")
    ap.add_argument("--fps", type=float, default=30, help="capture fps (0=auto)")
    ap.add_argument("--backend", type=str, default="auto", help="auto|v4l2|avfoundation")
    ap.add_argument("--mjpg", action="store_true", help="force MJPG fourcc (useful on Ubuntu/V4L2)")

    # ✅ 표시만 스케일(측정은 원본 view 기준)
    ap.add_argument("--preview_scale", type=float, default=1.0)

    # ✅ bg 크기 mismatch 자동 리사이즈(편의 옵션)
    ap.add_argument("--auto_resize_bg", action="store_true")

    args = ap.parse_args()

    corners_path = Path(args.corners)
    bg_path = Path(args.bg)

    corners = load_corners(corners_path)
    if corners is None:
        print(f"[WARN] corners not found/invalid -> RAW mode. ({corners_path.resolve()})")

    cap = open_capture(args.cam, args.backend)
    if not cap.isOpened():
        raise RuntimeError(f"Camera open failed. Try --cam 1 or 2 (current {args.cam})")

    req_w = args.width if args.width > 0 else None
    req_h = args.height if args.height > 0 else None
    req_fps = args.fps if args.fps and args.fps > 0 else None
    apply_capture_settings(cap, req_w, req_h, req_fps, args.mjpg)

    out_w = args.warp_w if args.warp_w > 0 else None
    out_h = args.warp_h if args.warp_h > 0 else None

    # 배경 로드
    bg_gray = None
    bg_shape = None
    if bg_path.exists():
        bg = cv2.imread(str(bg_path))
        if bg is not None:
            bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            bg_shape = bg_gray.shape
            print(f"[OK] Loaded background: {bg_path.resolve()}  shape={bg_gray.shape[::-1]}")
        else:
            print(f"[WARN] bg exists but failed to read: {bg_path.resolve()}")

    target_area = None
    over_count = 0
    area_hist = deque(maxlen=max(1, args.smooth))

    print("\nKeys:")
    print("  b : capture background (empty pan) -> save bg.png")
    print("  t : set target area = current measured area")
    print("  + / - : adjust target area (±200 px)")
    print("  r : reset target area (unset)")
    print("  q or ESC : quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERR] Frame read failed.")
            break

        # 1) view 생성 (warp 우선, 실패시 RAW)
        view, mode = build_view(frame, corners, out_w, out_h)

        gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)

        # 2) bg 매칭/차영상
        valid_bg = (bg_gray is not None)
        if valid_bg and bg_gray.shape != gray.shape:
            if args.auto_resize_bg:
                bg_gray_use = cv2.resize(bg_gray, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
                valid_bg = True
            else:
                valid_bg = False
        else:
            bg_gray_use = bg_gray

        if not valid_bg:
            mask = np.zeros_like(gray, dtype=np.uint8)
            area_px = 0
            contour = None
            reached = False
            area_hist.clear()
            over_count = 0
        else:
            diff = cv2.absdiff(gray, bg_gray_use)
            _, mask = cv2.threshold(diff, args.diff_thresh, 255, cv2.THRESH_BINARY)

            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

            contour, _ = largest_component(mask)
            area_px = int(cv2.countNonZero(mask))

            if area_px < args.min_area:
                area_px = 0
                contour = None

            area_hist.append(area_px)
            area_smooth = int(sum(area_hist) / len(area_hist))

            reached = False
            if target_area is not None and area_smooth >= target_area:
                over_count += 1
                if over_count >= args.stable_frames:
                    reached = True
            else:
                over_count = 0

            area_px = area_smooth

        # 3) 시각화
        vis = view.copy()
        if contour is not None:
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)

        line1 = f"MODE: {mode} | view: {vis.shape[1]}x{vis.shape[0]} | area(px): {area_px}"
        if target_area is None:
            line2 = "target: (unset) | press 't' to set target from current area"
        else:
            line2 = f"target: {target_area} | over_count: {over_count}/{args.stable_frames} | reached: {reached}"

        cv2.putText(vis, line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(vis, line2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        if bg_gray is None:
            cv2.putText(vis, "No bg. Press 'b' with empty pan to capture bg.",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        elif bg_gray is not None and bg_gray.shape != gray.shape and not args.auto_resize_bg:
            cv2.putText(vis,
                        f"BG mismatch! bg={bg_shape[::-1]} view={gray.shape[::-1]} (use --auto_resize_bg or recapture 'b')",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # display scale (only)
        if args.preview_scale != 1.0:
            disp = cv2.resize(vis, None, fx=args.preview_scale, fy=args.preview_scale, interpolation=cv2.INTER_AREA)
            disp_mask = cv2.resize(mask, None, fx=args.preview_scale, fy=args.preview_scale,
                                   interpolation=cv2.INTER_NEAREST)
        else:
            disp = vis
            disp_mask = mask

        cv2.imshow("view (for measurement)", disp)
        cv2.imshow("mask", disp_mask)

        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), 27]:
            break

        elif key == ord("b"):
            # 현재 view(=warp 적용된 화면)를 bg로 저장
            bg = view.copy()
            cv2.imwrite(str(bg_path), bg)
            bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            bg_shape = bg_gray.shape
            over_count = 0
            area_hist.clear()
            print(f"[OK] Captured background -> {bg_path.resolve()}  shape={bg_gray.shape[::-1]}")
            print("     (Tip) 꼭 '빈 팬' 상태에서 누르세요.")

        elif key == ord("t"):
            if bg_gray is None:
                print("[WARN] Set bg first with 'b' (empty pan).")
            else:
                target_area = int(max(0, area_px))
                over_count = 0
                print(f"[OK] target_area set to {target_area}")

        elif key == ord("r"):
            target_area = None
            over_count = 0
            print("[OK] target_area reset (unset)")

        elif key in [ord("+"), ord("=")]:
            if target_area is not None:
                target_area += 200
                print(f"[OK] target_area -> {target_area}")

        elif key in [ord("-"), ord("_")]:
            if target_area is not None:
                target_area = max(0, target_area - 200)
                print(f"[OK] target_area -> {target_area}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
