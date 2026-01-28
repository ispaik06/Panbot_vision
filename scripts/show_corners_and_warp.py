# scripts/show_corners_and_warp.py
import cv2
import numpy as np
import json
from pathlib import Path
import argparse
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CORNERS = PROJECT_ROOT / "calibration" / "corners.json"


def load_corners(path: Path):
    if not path.exists():
        print(f"[ERROR] corners file not found / 코너 파일이 없습니다: {path.resolve()}")
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR] failed to read/parse json / json 읽기 실패: {e}")
        return None

    pts = data.get("points", None)
    if not pts or len(pts) != 4:
        print("[ERROR] invalid format / 포맷 오류. Expect/기대: {'points': [{'x':..,'y':..}, ... x4]}")
        return None

    corners = []
    for i, p in enumerate(pts):
        if "x" not in p or "y" not in p:
            print(f"[ERROR] point {i} missing x/y keys / x,y 누락: {p}")
            return None
        corners.append((int(p["x"]), int(p["y"])))

    # TL, TR, BR, BL
    return corners


def compute_warp(frame, corners_xy, out_w=None, out_h=None):
    """
    corners_xy: list[(x,y)] in TL,TR,BR,BL order
    """
    corners = np.array(corners_xy, dtype=np.float32)
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


def open_capture(cam_index: int, backend: str):
    """
    backend:
      - "auto": OpenCV default
      - "v4l2": Linux V4L2
      - "avfoundation": macOS AVFoundation
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
    print(f"[Camera] requested / 요청: {width or 'auto'}x{height or 'auto'} @ {fps or 'auto'}fps, mjpg={mjpg}")
    print(f"[Camera] actual    / 실제: {actual_w}x{actual_h} @ {actual_fps:.2f}fps")

    return actual_w, actual_h


def maybe_scale(img, scale: float):
    if scale is None or abs(scale - 1.0) < 1e-6:
        return img
    if scale <= 0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index")

    ap.add_argument("--corners", type=str, default=str(DEFAULT_CORNERS), help="path to corners.json")

    # capture settings
    ap.add_argument("--width", type=int, default=1280, help="capture width (0=auto)")
    ap.add_argument("--height", type=int, default=720, help="capture height (0=auto)")
    ap.add_argument("--fps", type=float, default=30, help="capture fps (0=auto)")
    ap.add_argument("--mjpg", action="store_true", help="force MJPG fourcc (useful on Ubuntu/V4L2)")
    ap.add_argument("--backend", type=str, default="auto", help="auto|v4l2|avfoundation")

    # warp output (optional)
    ap.add_argument("--warp_w", type=int, default=0, help="warp output width (0=auto)")
    ap.add_argument("--warp_h", type=int, default=0, help="warp output height (0=auto)")

    # display scale
    ap.add_argument("--preview_scale", type=float, default=1.0, help="display scale only (e.g., 0.5, 1.0, 1.5)")

    args = ap.parse_args()

    corners_path = Path(args.corners)
    corners = load_corners(corners_path)
    if corners is None:
        print("[EXIT] corners.json missing or invalid / corners.json이 없거나 잘못되었습니다. 종료합니다.")
        sys.exit(1)

    labels = ["TL", "TR", "BR", "BL"]

    req_w = args.width if args.width > 0 else None
    req_h = args.height if args.height > 0 else None
    req_fps = args.fps if args.fps and args.fps > 0 else None

    out_w = args.warp_w if args.warp_w > 0 else None
    out_h = args.warp_h if args.warp_h > 0 else None

    cap = open_capture(args.cam, args.backend)
    if not cap.isOpened():
        print(f"[ERROR] camera open failed / 카메라 열기 실패. Try --cam 1 or 2 (current: {args.cam})")
        sys.exit(1)

    actual_w, actual_h = apply_capture_settings(cap, req_w, req_h, req_fps, args.mjpg)

    print(f"[OK] Loaded corners / 코너 로드 완료: {corners_path.resolve()}")
    print("[INFO] Windows / 창:")
    print("  - 'Corners Overlay (RAW)'  : raw frame + corner overlay / 원본+코너 표시")
    print("  - 'Top-View (Warped)'      : warped top-view / 워프 결과")
    print("Press 'q' or ESC to quit / 종료: q 또는 ESC")

    # quick bounds check
    out_of_bounds = [(x, y) for (x, y) in corners if x < 0 or y < 0 or x >= actual_w or y >= actual_h]
    if out_of_bounds:
        print("[WARN] Some corners are out of current frame bounds! / 코너가 화면 밖입니다!")
        print(f"       frame={actual_w}x{actual_h}, out_of_bounds={out_of_bounds}")
        print("       -> Match capture resolution used to create corners.json / corners.json 만든 해상도와 지금 해상도를 맞추세요 (--width/--height)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] frame read failed / 프레임 읽기 실패.")
            break

        # --- RAW overlay window ---
        raw_vis = frame.copy()

        for i, (x, y) in enumerate(corners):
            cv2.circle(raw_vis, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(raw_vis, labels[i], (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(raw_vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.putText(raw_vis, f"RAW {actual_w}x{actual_h}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        raw_disp = maybe_scale(raw_vis, args.preview_scale)
        cv2.imshow("Corners Overlay (RAW)", raw_disp)

        # --- Warped window ---
        warped, _, (wg, hg) = compute_warp(frame, corners, out_w=out_w, out_h=out_h)
        if warped is None:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(blank, f"Warp failed ({wg}x{hg})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            warped_disp = maybe_scale(blank, args.preview_scale)
            cv2.imshow("Top-View (Warped)", warped_disp)
        else:
            wtxt = f"WARP {warped.shape[1]}x{warped.shape[0]}"
            cv2.putText(warped, wtxt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            warped_disp = maybe_scale(warped, args.preview_scale)
            cv2.imshow("Top-View (Warped)", warped_disp)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
