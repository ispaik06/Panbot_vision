import cv2
import numpy as np
import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CORNERS = PROJECT_ROOT / "calibration" / "corners.json"

WINDOW = "Click corners: TL->TR->BR->BL | s=save, r=reset, q=quit"
points = []  # user clicks in TL, TR, BR, BL order

preview_scale = 1.0


def compute_warp(frame, corners, out_w=None, out_h=None):
    tl, tr, br, bl = corners

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    # 너무 작으면 워프가 깨짐
    if maxW < 50 or maxH < 50:
        return None, None, (maxW, maxH)

    # 출력 크기 고정 옵션
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


def on_mouse(event, x, y, flags, param):
    global points, preview_scale

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # disp 좌표 -> 원본 frame 좌표로 변환
    if preview_scale and abs(preview_scale - 1.0) > 1e-6:
        x0 = int(round(x / preview_scale))
        y0 = int(round(y / preview_scale))
    else:
        x0, y0 = x, y

    if len(points) < 4:
        points.append((x0, y0))
        labels = ["TL", "TR", "BR", "BL"]
        print(f"{labels[len(points)-1]} = ({x0}, {y0})")
    else:
        print("Already 4 points. Press 'r' to reset.")


def save_corners(out_path: Path, pts):
    data = {"points": [{"x": int(x), "y": int(y)} for (x, y) in pts]}
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[OK] Saved corners -> {out_path.resolve()}")


def open_capture(cam_index: int, backend: str | None):
    """
    backend:
      - None / "auto": OpenCV default
      - "v4l2": Linux V4L2
      - "avfoundation": macOS AVFoundation
    """
    if backend is None or backend.lower() == "auto":
        return cv2.VideoCapture(cam_index)

    b = backend.lower()
    if b == "v4l2":
        return cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if b == "avfoundation":
        return cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)

    raise ValueError(f"Unknown backend: {backend} (use auto|v4l2|avfoundation)")


def apply_capture_settings(cap, width, height, fps, mjpg: bool):
    # MJPG는 주로 Linux(V4L2)에서 해상도/FPS 협상에 도움이 됨
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, float(fps))

    # 실제 적용값 출력
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    backend_name = ""
    try:
        backend_name = cap.getBackendName()
    except Exception:
        backend_name = "(unknown)"

    print(f"[Camera] backend={backend_name}")
    print(f"[Camera] requested: {width or 'auto'}x{height or 'auto'} @ {fps or 'auto'}fps, mjpg={mjpg}")
    print(f"[Camera] actual:    {actual_w}x{actual_h} @ {actual_fps:.2f}fps")


def main():
    global points

    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index (0,1,2,...)")

    # ✅ 추가: 캡처 설정 고정 옵션
    ap.add_argument("--width", type=int, default=1280, help="capture width (0=auto)")
    ap.add_argument("--height", type=int, default=720, help="capture height (0=auto)")
    ap.add_argument("--fps", type=float, default=30, help="capture fps (0=auto)")
    ap.add_argument("--mjpg", action="store_true", help="force MJPG fourcc (useful on Ubuntu/V4L2)")
    ap.add_argument("--backend", type=str, default="auto", help="auto|v4l2|avfoundation")

    # 워프 출력 설정
    ap.add_argument("--warp_w", type=int, default=0, help="warp output width (0=auto)")
    ap.add_argument("--warp_h", type=int, default=0, help="warp output height (0=auto)")
    ap.add_argument("--out", type=str, default=str(DEFAULT_CORNERS), help="output json path")

    # ✅ 추가: 화면 표시만 확대/축소(좌표는 원본 프레임 기준)
    ap.add_argument("--preview_scale", type=float, default=1.0, help="preview scale for display only (e.g., 0.75, 1.0, 1.5)")

    args = ap.parse_args()

    global preview_scale
    preview_scale = float(args.preview_scale)
    if preview_scale <= 0:
        preview_scale = 1.0


    out_w = args.warp_w if args.warp_w > 0 else None
    out_h = args.warp_h if args.warp_h > 0 else None
    out_path = Path(args.out)

    req_w = args.width if args.width > 0 else None
    req_h = args.height if args.height > 0 else None
    req_fps = args.fps if args.fps and args.fps > 0 else None

    cap = open_capture(args.cam, args.backend)
    if not cap.isOpened():
        raise RuntimeError(f"Camera open failed. Try --cam 1 or 2. (current: {args.cam})")

    apply_capture_settings(cap, req_w, req_h, req_fps, args.mjpg)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    print("Click corners in order: TL -> TR -> BR -> BL")
    print("Keys: s=save corners.json, r=reset points, q/ESC=quit")
    if out_w or out_h:
        print(f"Warp output fixed to: {out_w if out_w else 'auto'} x {out_h if out_h else 'auto'}")
    print(f"Will save to: {out_path.resolve()}")
    if args.preview_scale != 1.0:
        print(f"[Preview] scale={args.preview_scale} (display only, saved coords are in original frame pixels)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed.")
            break

        vis = frame.copy()

        labels = ["TL", "TR", "BR", "BL"]
        for i, (x, y) in enumerate(points):
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(vis, labels[i], (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if len(points) == 4:
            corners = np.array(points, dtype=np.float32)  # TL,TR,BR,BL
            poly = corners.astype(int).reshape((-1, 1, 2))
            cv2.polylines(vis, [poly], True, (0, 255, 0), 2)

            warped, _, (w, h) = compute_warp(frame, corners, out_w=out_w, out_h=out_h)
            cv2.putText(vis, f"warp size: {w}x{h}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if warped is not None:
                warped_disp = warped
                if preview_scale and abs(preview_scale - 1.0) > 1e-6:
                    warped_disp = cv2.resize(
                        warped,
                        None,
                        fx=preview_scale,
                        fy=preview_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow("Top-View (Warped)", warped_disp)
            else:
                blank = np.zeros((200, 400, 3), dtype=np.uint8)
                if preview_scale and abs(preview_scale - 1.0) > 1e-6:
                    blank = cv2.resize(
                        blank,
                        None,
                        fx=preview_scale,
                        fy=preview_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow("Top-View (Warped)", blank)


        # ✅ preview_scale 적용 (표시만 변경)
        if args.preview_scale != 1.0:
            disp = cv2.resize(vis, None, fx=args.preview_scale, fy=args.preview_scale, interpolation=cv2.INTER_AREA)
        else:
            disp = vis

        cv2.imshow(WINDOW, disp)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord('r'):
            points = []
            print("Reset points.")
        elif key == ord('s'):
            if len(points) != 4:
                print("[WARN] Need 4 points (TL,TR,BR,BL) before saving.")
            else:
                save_corners(out_path, points)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
