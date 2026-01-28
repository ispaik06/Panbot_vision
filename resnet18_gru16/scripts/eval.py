# resnet18_gru16/scripts/eval.py

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# Optional TB
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_OK = True
except Exception:
    SummaryWriter = None
    _TB_OK = False

# Optional: plot confusion matrix to image for TB
import matplotlib.pyplot as plt


LABELS = ["not_ready", "almost_ready", "ready"]
LABEL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LABEL = {i: k for k, i in LABEL2ID.items()}


# -------------------------
# Dataset
# -------------------------
class SequenceIndexDataset(Dataset):
    """
    Reads index_*.csv produced by make_sequence_index.py
    columns: run_id,end_frame,label,frame_00..frame_15

    IMPORTANT:
    - frame_* paths are stored RELATIVE to Panbot_vision.
    - We assume you run eval from Panbot_vision directory.
    """
    def __init__(self, index_csv: Path, base_dir: Path, seq_len: int, image_size: int):
        self.index_csv = index_csv
        self.base_dir = base_dir
        self.seq_len = seq_len

        self.rows = []
        self.skipped_bad_rows = 0

        with index_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                label = (r.get("label") or "").strip()
                if label not in LABEL2ID:
                    self.skipped_bad_rows += 1
                    continue

                frames = []
                ok = True
                for i in range(seq_len):
                    key = f"frame_{i:02d}"
                    p = (r.get(key) or "").strip()
                    if not p:
                        ok = False
                        break

                    pp = Path(p)
                    if not pp.is_absolute():
                        pp = (self.base_dir / pp).resolve()

                    # Skip AppleDouble
                    if pp.name.startswith("._"):
                        ok = False
                        break

                    frames.append(pp)

                if not ok:
                    self.skipped_bad_rows += 1
                    continue

                self.rows.append((frames, LABEL2ID[label]))

        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        print(f"[DATASET] loaded={len(self.rows)}  skipped_bad_rows={self.skipped_bad_rows}  from={index_csv}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        frame_paths, y = self.rows[idx]
        imgs = []
        for p in frame_paths:
            im = Image.open(p).convert("RGB")
            imgs.append(self.tf(im))
        x = torch.stack(imgs, dim=0)  # (T, C, H, W)
        return x, torch.tensor(y, dtype=torch.long)


# -------------------------
# Model: ResNet18 + GRU
# -------------------------
class ResNet18GRU(nn.Module):
    def __init__(self, hidden_size: int = 256, num_layers: int = 1, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # -> (B, 512, 1, 1)
        self.feat_dim = 512

        self.gru = nn.GRU(
            input_size=self.feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feat = self.backbone(x).flatten(1)     # (B*T, 512)
        feat = feat.reshape(B, T, -1)          # (B, T, 512)
        out, _ = self.gru(feat)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1_from_cm(cm: np.ndarray) -> Dict[str, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    prec = tp / np.maximum(tp + fp, 1.0)
    rec  = tp / np.maximum(tp + fn, 1.0)
    f1   = 2 * prec * rec / np.maximum(prec + rec, 1e-12)

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_f1": f1.mean(),
        "acc": tp.sum() / np.maximum(cm.sum(), 1.0),
        "fp": fp,
        "fn": fn,
    }


def _plot_cm(cm: np.ndarray, labels: list, normalize: bool = False, title: str = "Confusion Matrix"):
    cm_plot = cm.astype(np.float64)
    if normalize:
        row_sum = np.maximum(cm_plot.sum(axis=1, keepdims=True), 1.0)
        cm_plot = cm_plot / row_sum

    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm_plot, interpolation="nearest")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    fmt = ".2f" if normalize else "d"
    thresh = (cm_plot.max() + cm_plot.min()) / 2.0 if cm_plot.size else 0.5
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val = cm_plot[i, j]
            ax.text(j, i, format(val, fmt),
                    ha="center", va="center",
                    color="white" if val > thresh else "black")

    ax.set_ylabel("True")
    ax.set_xlabel("Pred")
    fig.tight_layout()
    return fig


@torch.no_grad()
def evaluate(model, loader, device, amp: bool) -> Tuple[float, np.ndarray, Dict[str, np.ndarray]]:
    model.eval()

    total_loss = 0.0
    ys, ps = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if device.type == "cuda":
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)

        total_loss += float(loss.item()) * x.size(0)

        pred = torch.argmax(logits, dim=1)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    ys = np.concatenate(ys, axis=0) if ys else np.array([], dtype=np.int64)
    ps = np.concatenate(ps, axis=0) if ps else np.array([], dtype=np.int64)

    avg_loss = total_loss / max(len(loader.dataset), 1)
    cm = confusion_matrix(ys, ps, num_classes=len(LABELS))
    m = precision_recall_f1_from_cm(cm)
    return avg_loss, cm, m


def main():
    ap = argparse.ArgumentParser(description="Evaluate ResNet18+GRU16 classifier on test index CSV.")
    ap.add_argument("--root", type=str, default=None, help="resnet18_gru16 root (default: auto detect from scripts/..)")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt (best.pt or last.pt)")
    ap.add_argument("--test_csv", type=str, default="dataset/index_test.csv")
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=224)

    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="Use autocast for eval (CUDA only)")

    ap.add_argument("--tb", action="store_true", help="Log to TensorBoard under <out_dir>/tb_eval")
    ap.add_argument("--tb_dir", type=str, default="", help="Override TB dir (default: <out_dir>/tb_eval)")
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    default_root = script_path.parent.parent  # .../resnet18_gru16
    root = Path(args.root).resolve() if args.root else default_root

    # assume you run from Panbot_vision
    base_dir = Path.cwd().resolve()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    test_csv = (root / args.test_csv).resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    if not test_csv.exists():
        raise FileNotFoundError(f"test_csv not found: {test_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)
    print("[BASE_DIR for relative paths]", base_dir)
    print("[CKPT]", ckpt_path)
    print("[TEST_CSV]", test_csv)

    # Load checkpoint first (for model args)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Restore label mapping if saved
    label2id = ckpt.get("label2id", None)
    if isinstance(label2id, dict) and set(label2id.keys()) == set(LABEL2ID.keys()):
        # keep current LABELS order, but warn if mismatch IDs
        pass

    saved_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    hidden = int(saved_args.get("hidden", 256))
    gru_layers = int(saved_args.get("gru_layers", 1))

    # Build model (pretrained not needed for eval since weights load overrides)
    model = ResNet18GRU(
        hidden_size=hidden,
        num_layers=gru_layers,
        num_classes=len(LABELS),
        pretrained=False,
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)

    # Data
    ds_test = SequenceIndexDataset(test_csv, base_dir=base_dir, seq_len=args.seq_len, image_size=args.image_size)
    pin = (device.type == "cuda")
    dl_test = DataLoader(ds_test, batch_size=args.batch, shuffle=False,
                         num_workers=args.num_workers, pin_memory=pin)

    # TB
    writer = None
    out_dir = ckpt_path.parent  # usually runs/.../
    if args.tb:
        if not _TB_OK:
            raise RuntimeError("TensorBoard is not available. Install with: pip install tensorboard")
        tb_dir = Path(args.tb_dir).resolve() if args.tb_dir else (out_dir / "tb_eval")
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        print("[TB] logging to:", tb_dir)

    # Eval
    loss, cm, m = evaluate(model, dl_test, device, amp=(args.amp and device.type == "cuda"))

    print("\n[TEST]")
    print(f"  loss={loss:.4f}  acc={m['acc']:.3f}  macro_f1={m['macro_f1']:.3f}")
    for i, name in enumerate(LABELS):
        print(f"  - {name:12s}  P={m['precision'][i]:.3f} R={m['recall'][i]:.3f} "
              f"F1={m['f1'][i]:.3f}  FP={int(m['fp'][i])} FN={int(m['fn'][i])}")

    print("\n[CONFUSION_MATRIX] rows=true, cols=pred")
    print(cm)

    # TB logging
    if writer is not None:
        writer.add_scalar("test/loss", loss, 0)
        writer.add_scalar("test/acc", float(m["acc"]), 0)
        writer.add_scalar("test/macro_f1", float(m["macro_f1"]), 0)
        for i, name in enumerate(LABELS):
            writer.add_scalar(f"test/{name}/precision", float(m["precision"][i]), 0)
            writer.add_scalar(f"test/{name}/recall", float(m["recall"][i]), 0)
            writer.add_scalar(f"test/{name}/f1", float(m["f1"][i]), 0)
            writer.add_scalar(f"test/{name}/fp", float(m["fp"][i]), 0)
            writer.add_scalar(f"test/{name}/fn", float(m["fn"][i]), 0)

        fig1 = _plot_cm(cm, LABELS, normalize=False, title="Confusion Matrix (Counts)")
        writer.add_figure("test/confusion_matrix_counts", fig1, global_step=0)
        plt.close(fig1)

        fig2 = _plot_cm(cm, LABELS, normalize=True, title="Confusion Matrix (Row-normalized)")
        writer.add_figure("test/confusion_matrix_norm", fig2, global_step=0)
        plt.close(fig2)

        writer.flush()
        writer.close()

    print("\n[DONE]")


if __name__ == "__main__":
    main()
