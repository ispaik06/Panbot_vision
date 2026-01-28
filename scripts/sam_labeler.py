# scripts/sam_labeler.py
import cv2
import numpy as np
import argparse
from pathlib import Path
import torch
import re

# ---- paths ----
SCRIPT_DIR = Path(__file__).resolve().parent         # .../scripts
PROJECT_ROOT = SCRIPT_DIR.parent                     # .../pancake_vision
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "images"
DEFAULT_MASKS_DIR = PROJECT_ROOT / "masks"
DEFAULT_CKPT_DIR = PROJECT_ROOT / "model"
DEFAULT_CKPT_VITB = DEFAULT_CKPT_DIR / "sam_vit_b_01ec64.pth"
DEFAULT_CKPT_VITL = DEFAULT_CKPT_DIR / "sam_vit_l_0b3195.pth"
DEFAULT_CKPT_VITH = DEFAULT_CKPT_DIR / "sam_vit_h_4b8939.pth"

# SAM
from segment_anything import sam_model_registry, SamPredictor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def extract_trailing_int(stem: str):
    """Extract trailing integer from filename stem.
    e.g., 'img_000019' -> 19, 'frame12' -> 12, 'abc' -> None
    """
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else None


def list_images(images_dir: Path):
    paths = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)

    def sort_key(p: Path):
        n = extract_trailing_int(p.stem)
        # 번호가 있으면 번호로 우선 정렬, 없으면 이름 정렬
        return (0, n, p.name) if n is not None else (1, 0, p.name)

    return sorted(paths, key=sort_key)


def overlay_mask(image_bgr, mask_bool, alpha=0.45):
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    mask_color = np.zeros_like(image_bgr)
    mask_color[:, :, 1] = mask_u8  # green channel
    return cv2.addWeighted(image_bgr, 1.0, mask_color, alpha, 0)


def draw_points(vis, points, labels):
    for (x, y), lab in zip(points, labels):
        if lab == 1:
            color = (0, 255, 0)   # positive: green
            text = "+"
        else:
            color = (0, 0, 255)   # negative: red
            text = "-"
        cv2.circle(vis, (int(x), int(y)), 6, color, -1)
        cv2.putText(
            vis,
            text,
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )


def pick_device(device_arg: str):
    d = (device_arg or "auto").lower()
    if d != "auto":
        return d

    # auto: prefer cuda -> mps -> cpu
    if torch.cuda.is_available():
        return "cuda"
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def default_checkpoint_for(model_type: str):
    t = model_type.lower()
    if t == "vit_b":
        return DEFAULT_CKPT_VITB
    if t == "vit_l":
        return DEFAULT_CKPT_VITL
    if t == "vit_h":
        return DEFAULT_CKPT_VITH
    return DEFAULT_CKPT_VITB


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def resolve_index_by_start(img_paths, start_arg: int):
    """Interpret --start as:
    1) filename number (img_000019 -> 19) if match exists
    2) otherwise 0-based index
    """
    num_to_idx = {}
    for i, p in enumerate(img_paths):
        n = extract_trailing_int(p.stem)
        if n is not None and n not in num_to_idx:
            num_to_idx[n] = i

    if start_arg in num_to_idx:
        return num_to_idx[start_arg]

    return max(0, min(start_arg, len(img_paths) - 1))


def resolve_index_by_end(img_paths, end_arg: int):
    """Interpret --end as:
    -1 -> last image
    1) filename number if match exists
    2) otherwise 0-based index
    """
    if end_arg is None or end_arg < 0:
        return len(img_paths) - 1

    num_to_idx = {}
    for i, p in enumerate(img_paths):
        n = extract_trailing_int(p.stem)
        if n is not None and n not in num_to_idx:
            num_to_idx[n] = i

    if end_arg in num_to_idx:
        return num_to_idx[end_arg]

    return max(0, min(end_arg, len(img_paths) - 1))


def resize_preview(img_bgr, preview_scale: float):
    if abs(preview_scale - 1.0) < 1e-8:
        return img_bgr
    h, w = img_bgr.shape[:2]
    new_w = max(1, int(round(w * preview_scale)))
    new_h = max(1, int(round(h * preview_scale)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def resize_mask_preview(mask_bool, out_w: int, out_h: int):
    """Resize boolean mask to preview size using nearest neighbor."""
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    mask_rs = cv2.resize(mask_u8, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return (mask_rs > 127)


def scale_points_for_preview(points_xy, preview_scale: float):
    if abs(preview_scale - 1.0) < 1e-8:
        return points_xy
    return [(x * preview_scale, y * preview_scale) for (x, y) in points_xy]


def clamp_point(x: float, y: float, w: int, h: int):
    x = max(0, min(int(round(x)), w - 1))
    y = max(0, min(int(round(y)), h - 1))
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, default=str(DEFAULT_IMAGES_DIR), help="input images directory")
    ap.add_argument("--masks_dir", type=str, default=str(DEFAULT_MASKS_DIR), help="output masks directory")

    ap.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--checkpoint", type=str, default="", help="path to .pth checkpoint (empty=auto by model_type)")
    ap.add_argument("--device", type=str, default="auto", help="auto|cuda|mps|cpu")

    ap.add_argument(
        "--start",
        type=int,
        default=0,
        help="start index OR image number in filename (e.g., img_000019 -> 19)",
    )
    ap.add_argument(
        "--end",
        type=int,
        default=-1,
        help="end index OR image number in filename (inclusive). -1 means last image",
    )

    ap.add_argument("--alpha", type=float, default=0.45, help="mask overlay alpha")
    ap.add_argument("--save_binary_png", action="store_true", help="save mask as 0/255 png (recommended)")

    # ✅ 미리보기 스케일 (표시만 축소/확대, SAM은 원본 해상도 기준으로 동작)
    ap.add_argument(
        "--preview_scale",
        type=float,
        default=1.0,
        help="preview scale for display only (e.g., 0.5 shows half-size). Points are mapped to original image.",
    )

    args = ap.parse_args()

    if args.preview_scale <= 0:
        raise ValueError("--preview_scale must be > 0")

    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    ensure_dir(masks_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir.resolve()}")

    img_paths = list_images(images_dir)
    if not img_paths:
        raise RuntimeError(f"No images found in: {images_dir.resolve()}")

    device = pick_device(args.device)

    ckpt = Path(args.checkpoint) if args.checkpoint else default_checkpoint_for(args.model_type)
    if not ckpt.exists():
        print(f"[ERROR] checkpoint not found: {ckpt.resolve()}")
        print("Download checkpoint into PROJECT_ROOT/model/ (recommended). Example:")
        print("  mkdir -p model")
        print("  curl -L -o model/sam_vit_b_01ec64.pth \\")
        print("    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        raise SystemExit(1)

    print(f"[OK] images_dir     : {images_dir.resolve()}")
    print(f"[OK] masks_dir      : {masks_dir.resolve()}")
    print(f"[OK] model_type     : {args.model_type}")
    print(f"[OK] checkpoint     : {ckpt.resolve()}")
    print(f"[OK] device         : {device}")
    print(f"[OK] preview_scale  : {args.preview_scale}")
    print("\nMouse:")
    print("  Left click  = (+) batter / foreground")
    print("  Right click = (-) background (pan/table)")
    print("\nKeys:")
    print("  c = clear points")
    print("  s = save mask (and update saved-mask window)")
    print("  n = next image")
    print("  p = prev image")
    print("  q / ESC = quit\n")

    # Load SAM
    sam = sam_model_registry[args.model_type](checkpoint=str(ckpt))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Range resolve
    start_idx = resolve_index_by_start(img_paths, args.start)
    end_idx = resolve_index_by_end(img_paths, args.end)
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx  # auto-swap

    idx = start_idx
    points = []          # list[(x,y)] in ORIGINAL image coordinates
    labels = []          # list[int] 1/0
    current_mask = None  # bool (H,W) in ORIGINAL size

    WIN_MAIN = "SAM Labeler | LMB:+ RMB:- | c=clear s=save n=next p=prev q=quit"
    WIN_SAVED = "Saved Mask (file from masks_dir)"

    # 현재 이미지 원본 크기(마우스 좌표 변환에 필요)
    cur_w = None
    cur_h = None

    def on_mouse(event, x, y, flags, param):
        nonlocal points, labels, cur_w, cur_h
        if cur_w is None or cur_h is None:
            return

        # (preview 좌표) -> (original 좌표)
        ox = x / args.preview_scale
        oy = y / args.preview_scale
        ox, oy = clamp_point(ox, oy, cur_w, cur_h)

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((ox, oy))
            labels.append(1)
            print(f"[+] preview({x},{y}) -> orig({ox},{oy})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((ox, oy))
            labels.append(0)
            print(f"[-] preview({x},{y}) -> orig({ox},{oy})")

    # ✅ 창 2개를 시작부터 계속 띄워놓기
    cv2.namedWindow(WIN_MAIN)
    cv2.setMouseCallback(WIN_MAIN, on_mouse)

    cv2.namedWindow(WIN_SAVED, cv2.WINDOW_NORMAL)

    # 2번째 창: 같은 파일이면 매번 다시 읽지 않도록 캐시
    saved_cache = {"path": None, "mtime": None}

    def show_saved_mask_for_image(img_path: Path, img_bgr_original):
        """Show masks_dir/{img_stem}.png file itself in WIN_SAVED (scaled by preview_scale).
        If not saved, show a black canvas with the SAME SIZE as the main preview image.
        """
        mask_path = masks_dir / f"{img_path.stem}.png"

        # ✅ 현재 메인 창과 동일한 preview 크기 계산
        preview_img = resize_preview(img_bgr_original, args.preview_scale)
        ph, pw = preview_img.shape[:2]

        # 파일 없으면 안내 화면 (메인 preview 크기와 동일한 검은 캔버스)
        if not mask_path.exists():
            if saved_cache["path"] == str(mask_path) and saved_cache["mtime"] is None:
                return
            saved_cache["path"], saved_cache["mtime"] = str(mask_path), None

            canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
            cv2.resizeWindow(WIN_SAVED, pw, ph)
            cv2.putText(
                canvas,
                f"MASK NOT SAVED: {mask_path.name}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )
            cv2.imshow(WIN_SAVED, canvas)
            return

        mtime = mask_path.stat().st_mtime
        if saved_cache["path"] == str(mask_path) and saved_cache["mtime"] == mtime:
            return
        saved_cache["path"], saved_cache["mtime"] = str(mask_path), mtime

        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
            cv2.resizeWindow(WIN_SAVED, pw, ph)
            cv2.putText(
                canvas,
                f"FAILED TO READ: {mask_path.name}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )
            cv2.imshow(WIN_SAVED, canvas)
            return

        # 저장본이 0/1 이든 0/255 이든 보기 좋게 0/255로 변환
        if m.max() <= 1:
            m_vis = (m.astype(np.uint8) * 255)
        else:
            m_vis = m

        vis = cv2.cvtColor(m_vis, cv2.COLOR_GRAY2BGR)

        # ✅ 마스크도 preview_scale 반영 (NN)
        if abs(args.preview_scale - 1.0) > 1e-8:
            h, w = vis.shape[:2]
            new_w = max(1, int(round(w * args.preview_scale)))
            new_h = max(1, int(round(h * args.preview_scale)))
            vis = cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        ph2, pw2 = vis.shape[:2]
        cv2.resizeWindow(WIN_SAVED, pw2, ph2)

        cv2.putText(
            vis,
            f"{mask_path.name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
        cv2.imshow(WIN_SAVED, vis)


    # 프로그램 시작 시 2번째 창에 안내 화면 먼저 띄움
    init_canvas = np.zeros((240, 900, 3), dtype=np.uint8)
    cv2.putText(init_canvas, "Saved mask file will be shown here (from masks_dir).",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(init_canvas, "Press 's' to save. Use n/p to move images.",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow(WIN_SAVED, init_canvas)

    def recompute_mask(img_bgr_original):
        nonlocal current_mask
        if len(points) == 0:
            current_mask = None
            return

        img_rgb = cv2.cvtColor(img_bgr_original, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)

        point_coords = np.array(points, dtype=np.float32)   # ORIGINAL coords
        point_labels = np.array(labels, dtype=np.int32)

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        current_mask = masks[best].astype(bool)
        
    def print_missing_masks_and_exit():
        missing = []
        for p in img_paths[start_idx : end_idx + 1]:
            mask_path = masks_dir / f"{p.stem}.png"
            if not mask_path.exists():
                missing.append(p.name)

        if missing:
            print("\n[NOT SAVED] Missing masks in the selected range:")
            for name in missing:
                print(f"  - {name}")
            print(f"[NOT SAVED] total: {len(missing)}\n")
        else:
            print("\n[OK] All masks are saved in the selected range.\n")


    while True:
        img_path = img_paths[idx]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] failed to read: {img_path}")
            points, labels, current_mask = [], [], None
            if idx < end_idx:
                idx += 1
                continue
            else:
                break

        # ✅ 현재 이미지에 해당하는 "저장된 마스크 파일"을 2번째 창에 항상 갱신
        show_saved_mask_for_image(img_path, img_bgr)

        # update current original size for mouse mapping
        cur_h, cur_w = img_bgr.shape[:2]

        # compute mask (if points exist) using ORIGINAL image
        recompute_mask(img_bgr)

        # build preview visualization (main window)
        vis = resize_preview(img_bgr, args.preview_scale)

        if current_mask is not None:
            ph, pw = vis.shape[:2]
            mask_prev = resize_mask_preview(current_mask, pw, ph)
            vis = overlay_mask(vis, mask_prev, alpha=args.alpha)

        # draw points on preview (scaled)
        pts_prev = scale_points_for_preview(points, args.preview_scale)
        draw_points(vis, pts_prev, labels)

        header = (
            f"{idx-start_idx+1}/{end_idx-start_idx+1}  {img_path.name}  "
            f"| points={len(points)} | range=[{start_idx}..{end_idx}] | scale={args.preview_scale:g}"
        )
        cv2.putText(vis, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out_path = masks_dir / f"{img_path.stem}.png"
        status = "MASK: exists" if out_path.exists() else "MASK: not saved"
        cv2.putText(vis, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        cv2.imshow(WIN_MAIN, vis)

        key = cv2.waitKey(30) & 0xFF

        if key in [ord("q"), 27]:
            print_missing_masks_and_exit()
            break

        elif key == ord("c"):
            points, labels, current_mask = [], [], None
            print("[OK] cleared points")

        elif key == ord("n"):
            points, labels, current_mask = [], [], None
            if idx < end_idx:
                idx += 1
            else:
                print("[INFO] reached end")

        elif key == ord("p"):
            points, labels, current_mask = [], [], None
            if idx > start_idx:
                idx -= 1
            else:
                print("[INFO] reached start")

        elif key == ord("s"):
            out_path = masks_dir / f"{img_path.stem}.png"

            # ✅ 점 없으면: 올블랙 저장
            if current_mask is None:
                if cur_h is None or cur_w is None:
                    print("[ERR] cannot save empty mask because image size is unknown")
                    continue
                out = np.zeros((cur_h, cur_w), dtype=np.uint8)  # 0 (검정)
                ok = cv2.imwrite(str(out_path), out)
                if ok:
                    print(f"[SAVE] (empty) {out_path}")
                    # ✅ 저장 직후 2번째 창 즉시 갱신
                    show_saved_mask_for_image(img_path, img_bgr)
                else:
                    print(f"[ERR] failed to save: {out_path}")
                continue

            # ✅ 마스크가 있으면 기존대로 저장
            if args.save_binary_png:
                out = (current_mask.astype(np.uint8) * 255)  # ORIGINAL size 저장
            else:
                out = current_mask.astype(np.uint8)          # 0/1

            ok = cv2.imwrite(str(out_path), out)
            if ok:
                print(f"[SAVE] {out_path}")
                # ✅ 저장 직후 2번째 창 즉시 갱신
                show_saved_mask_for_image(img_path, img_bgr)
            else:
                print(f"[ERR] failed to save: {out_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
