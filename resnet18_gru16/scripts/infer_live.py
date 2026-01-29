# resnet18_gru16/scripts/infer_live.py
"""
Live inference for bubble readiness (ResNet18 + GRU).

- Webcam stream
- Rolling buffer, sequence sampling matches training:
    seq_len=16, stride=6
  Required buffer length = (seq_len-1)*stride + 1

- Optional warp (recommended if trained on warped images):
    --corners calibration/corners.json

- Displays live window with predicted label + confidence.
- Temporal smoothing:
    - EMA over probabilities
    - require K consecutive "ready" before showing READY (ready_hold)

- preview_scale: downscale display window only (model input stays image_size)
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# -------------------------
# Labels (fallback)
# -------------------------
DEFAULT_LABELS = ["not_ready", "almost_ready", "ready"]


# -------------------------
# Model: ResNet18 + GRU
# -------------------------
class ResNet18GRU(nn.Module):
    def __init__(self, hidden_size: int = 256, num_layers: int = 1, num_classes: int = 3, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # (B,512,1,1)
        self.feat_dim = 512

        self.gru = nn.GRU(
            input_size=self.feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feat = self.backbone(x).flatten(1)      # (B*T,512)
        feat = feat.reshape(B, T, -1)           # (B,T,512)
        out, _ = self.gru(feat)                 # (B,T,H)
        last = out[:, -1, :]                    # (B,H)
        return self.head(last)                  # (B,K)


# -------------------------
# Warp utilities (optional)
# -------------------------
def load_corners(corners_path: Path) -> np.ndarray:
    data = json.loads(corners_path.read_text(encoding="utf-8"))
    pts = data.get("points", None)
    if not pts or len(pts) != 4:
        raise ValueError("corners.json must contain 'points' with 4 entries")
    corners = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)
    return corners


def compute_warp_size(corners: np.ndarray) -> Tuple[int, int]:
    tl, tr, br, bl = corners
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    return max(maxW, 2), max(maxH, 2)


def warp_frame(frame_bgr: np.ndarray, corners: np.ndarray, out_w: Optional[int], out_h: Optional[int]) -> np.ndarray:
    if out_w is None or out_h is None:
        w, h = compute_warp_size(corners)
        out_w = w if out_w is None else out_w
        out_h = h if out_h is None else out_h

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(frame_bgr, M, (int(out_w), int(out_h)), flags=cv2.INTER_LINEAR)


# -------------------------
# Preprocess (match train.py: ImageNet norm)
# -------------------------
def get_imagenet_norm(device: torch.device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return mean, std


def preprocess_frame_bgr(frame_bgr: np.ndarray, image_size: int, device: torch.device,
                         mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # BGR -> RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # resize to model input
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)

    # to tensor [0,1]
    x = torch.from_numpy(rgb).to(torch.float32) / 255.0    # (H,W,C)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)         # (1,C,H,W)

    # normalize
    x = (x - mean) / std
    return x


# -------------------------
# Camera open helpers
# -------------------------
def open_capture(index: int, backend: str) -> cv2.VideoCapture:
    b = (backend or "auto").lower()
    if b == "auto":
        return cv2.VideoCapture(index)
    if b == "v4l2":
        return cv2.VideoCapture(index, cv2.CAP_V4L2)
    if b == "gstreamer":
        return cv2.VideoCapture(index, cv2.CAP_GSTREAMER)
    raise ValueError("backend must be one of: auto | v4l2 | gstreamer")


def try_set_capture(cap: cv2.VideoCapture, width: int, height: int, fps: float, mjpg: bool):
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(fps))


def fourcc_to_str(fourcc: int) -> str:
    return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--checkpoint", type=str,
                    default="resnet18_gru16/runs/resnet18_gru16_cls/best.pt",
                    help="Path to best.pt or last.pt")

    # must match training
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=6)

    # live camera
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--backend", type=str, default="v4l2", help="auto|v4l2|gstreamer")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--mjpg", action="store_true")

    # display
    ap.add_argument("--preview_scale", type=float, default=1.0,
                    help="Downscale preview window only (e.g., 0.3). Model input unchanged.")
    ap.add_argument("--no_window", action="store_true")

    # warp (optional)
    ap.add_argument("--corners", type=str, default="",
                    help="Optional: corners.json for warp (use if trained on warped images)")
    ap.add_argument("--warp_w", type=int, default=0)
    ap.add_argument("--warp_h", type=int, default=0)

    # smoothing / decision
    ap.add_argument("--ema", type=float, default=0.7,
                    help="EMA factor for probabilities (0~1). Higher=more smooth")
    ap.add_argument("--ready_hold", type=int, default=3,
                    help="Require N consecutive 'ready' to show READY")
    ap.add_argument("--print_every", type=int, default=15,
                    help="Print prediction every N inferences (0=off)")

    # amp
    ap.add_argument("--amp", action="store_true", help="Use autocast for inference on CUDA")

    args = ap.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    # load checkpoint
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("Checkpoint format invalid. Expect dict with key 'model'.")

    # labels
    label2id: Dict[str, int] = ckpt.get("label2id", {name: i for i, name in enumerate(DEFAULT_LABELS)})
    id2label = {i: k for k, i in label2id.items()}
    num_classes = len(label2id)

    # model config (prefer checkpoint args)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    hidden = int(ckpt_args.get("hidden", 256))
    gru_layers = int(ckpt_args.get("gru_layers", 1))

    model = ResNet18GRU(hidden_size=hidden, num_layers=gru_layers, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    print("[CKPT]", ckpt_path)
    print("[MODEL] hidden=", hidden, "gru_layers=", gru_layers, "classes=", num_classes)
    print("[LABELS]", [id2label[i] for i in range(num_classes)])

    # warp (optional)
    corners = None
    warp_w = warp_h = None
    if args.corners.strip():
        corners_path = Path(args.corners).expanduser().resolve()
        corners = load_corners(corners_path)
        warp_w = args.warp_w if args.warp_w > 0 else None
        warp_h = args.warp_h if args.warp_h > 0 else None
        print("[WARP] ON corners=", corners_path, "size=", (warp_w or "auto"), "x", (warp_h or "auto"))
    else:
        print("[WARP] OFF")

    # rolling buffer
    need = (args.seq_len - 1) * args.stride + 1
    buf = deque(maxlen=need)
    print("[SEQ] seq_len=", args.seq_len, "stride=", args.stride, "buffer_need=", need)

    # smoothing
    ema_prob = None
    ready_id = label2id.get("ready", num_classes - 1)
    almost_id = label2id.get("almost_ready", max(0, num_classes - 2))
    ready_streak = 0
    infer_count = 0
    last_print = 0

    # norm tensors on correct device
    mean, std = get_imagenet_norm(device)

    # open camera
    cap = open_capture(args.cam, args.backend)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index={args.cam} backend={args.backend}")

    try_set_capture(cap, args.width, args.height, args.fps, args.mjpg)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    print(f"[CAM] {actual_w}x{actual_h} fps={actual_fps:.2f} fourcc={fourcc_to_str(fourcc)} (requested fps={args.fps})")

    if not args.no_window:
        print("Press 'q' or ESC to quit.")
    else:
        print("[HEADLESS] no window (Ctrl+C to quit)")

    t0 = time.time()

    try:
        with torch.no_grad():
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[WARN] frame read fail")
                    continue

                # warp if enabled
                frame_proc = warp_frame(frame, corners, warp_w, warp_h) if corners is not None else frame
                buf.append(frame_proc)

                pred_label = "warming_up"
                conf = 0.0
                shown_id = None
                ema_lbl = None

                if len(buf) == need:
                    # sample indices for sequence
                    indices = [need - 1 - args.stride * (args.seq_len - 1 - i) for i in range(args.seq_len)]
                    frames = [buf[idx] for idx in indices]

                    xs = [preprocess_frame_bgr(f, args.image_size, device, mean, std) for f in frames]
                    x = torch.cat(xs, dim=0).unsqueeze(0)  # (1,T,C,H,W)

                    use_amp = bool(args.amp) and (device.type == "cuda")
                    if use_amp:
                        with torch.amp.autocast(device_type="cuda", enabled=True):
                            logits = model(x)
                    else:
                        logits = model(x)

                    prob = torch.softmax(logits, dim=1).float().cpu().numpy()[0]

                    # EMA smoothing
                    if ema_prob is None:
                        ema_prob = prob
                    else:
                        ema_prob = args.ema * ema_prob + (1.0 - args.ema) * prob

                    pred_id = int(np.argmax(ema_prob))
                    conf = float(ema_prob[pred_id])
                    ema_lbl = id2label.get(pred_id, str(pred_id))

                    # ready hold
                    if pred_id == ready_id:
                        ready_streak += 1
                    else:
                        ready_streak = 0

                    shown_id = pred_id
                    if pred_id == ready_id and ready_streak < max(args.ready_hold, 1):
                        shown_id = almost_id

                    pred_label = id2label.get(shown_id, str(shown_id))

                    infer_count += 1
                    if args.print_every > 0 and (infer_count - last_print) >= args.print_every:
                        last_print = infer_count
                        print(f"[PRED] ema={ema_lbl:12s} shown={pred_label:12s} conf={conf:.3f} ready_streak={ready_streak}")

                # preview window
                if not args.no_window:
                    vis = frame_proc.copy()

                    dt = time.time() - t0
                    fps_est = (infer_count / dt) if dt > 0 else 0.0

                    text1 = f"{pred_label}  conf={conf:.2f}  hold={ready_streak}/{args.ready_hold}"
                    text2 = f"buf={len(buf)}/{need}  infer={infer_count}  fps~{fps_est:.1f}"

                    cv2.putText(vis, text1, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    cv2.putText(vis, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    # preview_scale (display only)
                    s = float(args.preview_scale)
                    if s > 0 and abs(s - 1.0) > 1e-3:
                        new_w = max(1, int(vis.shape[1] * s))
                        new_h = max(1, int(vis.shape[0] * s))
                        vis = cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    cv2.imshow("infer_live (ResNet18+GRU)", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):  # q or ESC
                        break

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected. Exiting...")

    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
