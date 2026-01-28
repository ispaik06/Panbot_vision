# scripts/calibrate_from_image.py
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUT = PROJECT_ROOT / "calibration" / "corners.json"

WINDOW = "Calibration (Image) - Click TL->TR->BR->BL | s=save r=reset q=quit"

points = []          # points in ORIGINAL image coordinates
disp_scale = 1.0     # display scale (for BOTH windows)
orig_img = None      # original image (BGR)
orig_h = orig_w = 0


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def compute_warp(frame, corners, out_w=None, out_h=None):
    # corners: np.float32 (4,2) in TL,TR,BR,BL order
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


def save_corners(out_path: Path, pts):
    data = {"points": [{"x": int(x), "y": int(y)} for (x, y) in pts]}
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[OK] Saved corners -> {out_path.resolve()}")


def show_warp(warped_bgr, window_name="Warp Preview"):
    """Warp 창도 preview_scale 적용해서 보여주기"""
    global disp_scale
    if warped_bgr is None:
        return
    view = warped_bgr
    if disp_scale is not None and abs(disp_scale - 1.0) > 1e-6:
        new_w = int(view.shape[1] * disp_scale)
        new_h = int(view.shape[0] * disp_scale)
        if new_w >= 2 and new_h >= 2:
            view = cv2.resize(view, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imshow(window_name, view)


def on_mouse(event, x, y, flags, param):
    global points, disp_scale
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if len(points) >= 4:
        print("[INFO] Already 4 points. Press 'r' to reset.")
        return

    # x,y are DISPLAY coordinates -> convert to ORIGINAL coordinates
    ox = int(round(x / disp_scale))
    oy = int(round(y / disp_scale))

    # clamp
    ox = max(0, min(orig_w - 1, ox))
    oy = max(0, min(orig_h - 1, oy))

    points.append((ox, oy))
    labels = ["TL", "TR", "BR", "BL"]
    print(f"{labels[len(points)-1]} = ({ox}, {oy})  [clicked display=({x},{y}), scale={disp_scale}]")

    # show warp immediately when 4 points
    if len(points) == 4:
        corners = np.array(points, dtype=np.float32)
        warped, _, (w, h) = compute_warp(orig_img, corners, out_w=None, out_h=None)
        if warped is None:
            print(f"[WARN] Warp failed (auto size guess={w}x{h})")
        else:
            show_warp(warped, "Warp Preview (auto size)")


def main():
    global disp_scale, orig_img, orig_h, orig_w, points

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True, help="input image path (the exact image you want to calibrate on)")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="output corners.json path")

    ap.add_argument("--warp_w", type=int, default=0, help="warp output width for preview (0=auto)")
    ap.add_argument("--warp_h", type=int, default=0, help="warp output height for preview (0=auto)")

    ap.add_argument("--preview_scale", type=float, default=1.0,
                    help="display scale for BOTH windows (e.g., 0.5, 0.75, 1.0, 1.5). Click coords will be converted correctly.")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] image not found: {img_path.resolve()}")
        sys.exit(1)

    orig_img = cv2.imread(str(img_path))
    if orig_img is None:
        print(f"[ERROR] failed to read image: {img_path.resolve()}")
        sys.exit(1)

    orig_h, orig_w = orig_img.shape[:2]
    disp_scale = float(args.preview_scale) if args.preview_scale and args.preview_scale > 0 else 1.0

    out_path = Path(args.out)

    out_w = args.warp_w if args.warp_w > 0 else None
    out_h = args.warp_h if args.warp_h > 0 else None

    print(f"[OK] image : {img_path.resolve()}  ({orig_w}x{orig_h})")
    print(f"[OK] out   : {out_path.resolve()}")
    print("Click order: TL -> TR -> BR -> BL")
    print("Keys: s=save corners.json, r=reset points, q/ESC=quit")
    if disp_scale != 1.0:
        print(f"[Preview] scale={disp_scale} (applies to BOTH Raw & Warp windows; saved coords are ORIGINAL pixels)")
    if out_w or out_h:
        print(f"[Warp Preview] fixed output size = {out_w or 'auto'} x {out_h or 'auto'}")

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        vis = orig_img.copy()

        labels = ["TL", "TR", "BR", "BL"]
        for i, (x, y) in enumerate(points):
            cv2.circle(vis, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(vis, labels[i], (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if len(points) == 4:
            corners_i = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [corners_i], True, (0, 255, 0), 2)

            corners_f = np.array(points, dtype=np.float32)
            warped, _, (wg, hg) = compute_warp(orig_img, corners_f, out_w=out_w, out_h=out_h)
            if warped is not None:
                show_warp(warped, "Warp Preview")
            else:
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(blank, f"Warp failed ({wg}x{hg})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                show_warp(blank, "Warp Preview")

        # scale RAW display
        if abs(disp_scale - 1.0) > 1e-6:
            disp = cv2.resize(vis, None, fx=disp_scale, fy=disp_scale, interpolation=cv2.INTER_AREA)
        else:
            disp = vis

        cv2.imshow(WINDOW, disp)
        key = cv2.waitKey(20) & 0xFF

        if key in [ord("q"), 27]:
            break
        if key == ord("r"):
            points = []
            print("[INFO] Reset points.")
        if key == ord("s"):
            if len(points) != 4:
                print("[WARN] Need 4 points (TL,TR,BR,BL) before saving.")
            else:
                save_corners(out_path, points)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
