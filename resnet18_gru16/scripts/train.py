# resnet18_gru16/scripts/train.py
# - ResNet18 + GRU sequence classifier (3 labels)
# - Reads dataset/index_{train,val,test}.csv made by make_sequence_index.py
# - frame_* paths are stored RELATIVE to Panbot_vision (recommended)
# - Robust to AppleDouble (._*) and bad/corrupt images (skip unless --strict)
# - TensorBoard (--tb) and optional W&B (--wandb)
# - Saves checkpoints: last.pt, best.pt (best by val macro_f1)
# - ✅ Resume training from checkpoint (--resume)
# - ✅ Saves/restores AMP scaler state
# - Evaluates val every epoch + optional test after training (--eval_test)

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, UnidentifiedImageError

# optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

# optional wandb
try:
    import wandb
except Exception:
    wandb = None


# -------------------------
# Labels
# -------------------------
LABELS = ["not_ready", "almost_ready", "ready"]
LABEL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LABEL = {i: k for k, i in LABEL2ID.items()}


# -------------------------
# Utils: metrics
# -------------------------
@torch.no_grad()
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1_from_cm(cm: np.ndarray) -> Dict[str, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    prec = tp / np.maximum(tp + fp, 1.0)
    rec = tp / np.maximum(tp + fn, 1.0)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_f1": f1.mean(),
        "acc": tp.sum() / np.maximum(cm.sum(), 1.0),
        "fp": fp,
        "fn": fn,
    }


def cm_to_rgb_image(cm: np.ndarray) -> np.ndarray:
    """
    Confusion matrix -> HxWx3 uint8 image (simple grayscale heatmap, no matplotlib).
    """
    cm = cm.astype(np.float32)
    if cm.max() > 0:
        cm = cm / cm.max()
    img = (cm * 255.0).clip(0, 255).astype(np.uint8)      # (H,W)
    img = np.stack([img, img, img], axis=-1)              # (H,W,3)
    return img


# -------------------------
# Dataset
# -------------------------
class SequenceIndexDataset(Dataset):
    """
    Reads index_*.csv produced by make_sequence_index.py
    columns: run_id,end_frame,label,frame_00..frame_{T-1}

    Assumptions:
    - frame_* paths are stored RELATIVE to Panbot_vision (recommended).
    - You run this script from Panbot_vision (so base_dir = cwd).
    """
    def __init__(
        self,
        index_csv: Path,
        base_dir: Path,
        seq_len: int,
        image_size: int,
        strict: bool = False,
        skip_appledouble: bool = True,
    ):
        self.index_csv = index_csv
        self.base_dir = base_dir
        self.seq_len = seq_len
        self.strict = strict
        self.skip_appledouble = skip_appledouble

        self.rows: List[Tuple[List[Path], int]] = []
        bad_rows = 0

        with index_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            # 헤더가 이상하거나 key가 없을 때 바로 티나게
            fieldnames = set(reader.fieldnames or [])
            required = {"label"} | {f"frame_{i:02d}" for i in range(seq_len)}
            if not required.issubset(fieldnames):
                missing = sorted(list(required - fieldnames))
                raise ValueError(f"index csv missing columns: {missing}\nfile={index_csv}")

            for r in reader:
                label = (r.get("label") or "").strip()
                if label not in LABEL2ID:
                    bad_rows += 1
                    continue

                frames: List[Path] = []
                ok = True

                for i in range(seq_len):
                    key = f"frame_{i:02d}"
                    p = (r.get(key) or "").strip()
                    if not p:
                        ok = False
                        break

                    pp = Path(p)
                    if not pp.is_absolute():
                        pp = base_dir / pp  # do NOT resolve(strict=True), avoid crashing on missing

                    if skip_appledouble and pp.name.startswith("._"):
                        ok = False
                        break

                    frames.append(pp)

                if not ok:
                    bad_rows += 1
                    continue

                self.rows.append((frames, LABEL2ID[label]))

        print(f"[DATASET] loaded={len(self.rows)}  skipped_bad_rows={bad_rows}  from={index_csv}")

        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.rows)

    def _load_one(self, p: Path) -> torch.Tensor:
        im = Image.open(p).convert("RGB")
        return self.tf(im)

    def __getitem__(self, idx: int):
        if len(self.rows) == 0:
            raise RuntimeError(f"Empty dataset: {self.index_csv}")

        tries = 0
        cur = idx

        while True:
            frame_paths, y = self.rows[cur]
            imgs: List[torch.Tensor] = []
            try:
                for p in frame_paths:
                    if self.skip_appledouble and p.name.startswith("._"):
                        raise FileNotFoundError(f"AppleDouble file: {p}")
                    imgs.append(self._load_one(p))
                x = torch.stack(imgs, dim=0)  # (T,C,H,W)
                return x, torch.tensor(y, dtype=torch.long)
            except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
                if self.strict:
                    raise
                tries += 1
                cur = (cur + 1) % len(self.rows)
                if tries >= 20:
                    raise RuntimeError(f"Too many bad samples while fetching. Last error: {e}")


# -------------------------
# Model: ResNet18 + GRU
# -------------------------
class ResNet18GRU(nn.Module):
    def __init__(self, hidden_size: int = 256, num_layers: int = 1, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # -> (B,512,1,1)
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
        feat = self.backbone(x).flatten(1)   # (B*T,512)
        feat = feat.reshape(B, T, -1)        # (B,T,512)
        out, _ = self.gru(feat)              # (B,T,H)
        last = out[:, -1, :]
        return self.head(last)               # (B,num_classes)


# -------------------------
# Train / Eval
# -------------------------
def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    train: bool,
    log_every: int = 200,
):
    model.train() if train else model.eval()

    total_loss = 0.0
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []

    for step, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        use_amp = (scaler is not None) and (device.type == "cuda")
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.item()) * x.size(0)

        pred = torch.argmax(logits, dim=1)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

        if log_every > 0 and (step % log_every) == 0:
            avg = total_loss / max(step * loader.batch_size, 1)
            if device.type == "cuda":
                mem = torch.cuda.memory_allocated() / (1024**3)
                print(f"[{'TR' if train else 'VA'}] step {step:5d}/{len(loader)}  loss={avg:.4f}  gpu_mem={mem:.2f}GB")
            else:
                print(f"[{'TR' if train else 'VA'}] step {step:5d}/{len(loader)}  loss={avg:.4f}")

    ys_np = np.concatenate(ys, axis=0) if ys else np.array([], dtype=np.int64)
    ps_np = np.concatenate(ps, axis=0) if ps else np.array([], dtype=np.int64)
    avg_loss = total_loss / max(len(loader.dataset), 1)

    cm = confusion_matrix(ys_np, ps_np, num_classes=len(LABELS))
    m = precision_recall_f1_from_cm(cm)
    return avg_loss, cm, m


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    best_macro_f1: float,
    args: dict,
):
    ckpt = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_macro_f1": float(best_macro_f1),
        "label2id": LABEL2ID,
        "args": args,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, path)


def load_ckpt(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
):
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model"], strict=True)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scaler is not None and "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"[RESUME] scaler state load failed (ignored): {e}")

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_macro_f1 = float(ckpt.get("best_macro_f1", -1.0))

    return start_epoch, best_macro_f1, ckpt


def main():
    ap = argparse.ArgumentParser()

    # paths
    ap.add_argument("--root", type=str, default=None, help="resnet18_gru16 root (default: auto detect as scripts/..)")
    ap.add_argument("--train_csv", type=str, default="dataset/index_train.csv")
    ap.add_argument("--val_csv", type=str, default="dataset/index_val.csv")
    ap.add_argument("--test_csv", type=str, default="dataset/index_test.csv")
    ap.add_argument("--out_dir", type=str, default="runs/resnet18_gru16_cls")

    # data/model
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--gru_layers", type=int, default=1)
    ap.add_argument("--no_pretrained", action="store_true")

    # train hyperparams
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    # runtime
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--log_every", type=int, default=200, help="print progress every N steps (0=off)")

    # dataset robustness
    ap.add_argument("--strict", action="store_true", help="crash on missing/bad images instead of skipping")

    # logging
    ap.add_argument("--tb", action="store_true", help="Enable TensorBoard logging")
    ap.add_argument("--tb_dirname", type=str, default="tb", help="Subdir inside out_dir for tensorboard logs")
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", type=str, default="panbot_resnet18_gru16")
    ap.add_argument("--wandb_run_name", type=str, default="")

    # eval
    ap.add_argument("--eval_test", action="store_true", help="evaluate test set after training (uses best.pt)")

    # ✅ resume
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt to resume (e.g., runs/.../last.pt)")
    ap.add_argument("--resume_strict_args", action="store_true",
                    help="If set, checks that key training args match checkpoint (seq_len/image_size/hidden/gru_layers/labels).")

    args = ap.parse_args()

    # roots
    script_path = Path(__file__).resolve()
    default_root = script_path.parent.parent  # .../resnet18_gru16
    root = Path(args.root).resolve() if args.root else default_root

    # IMPORTANT: run from Panbot_vision
    base_dir = Path.cwd().resolve()

    train_csv = (root / args.train_csv).resolve()
    val_csv = (root / args.val_csv).resolve()
    test_csv = (root / args.test_csv).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # save config
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)
    print("[BASE_DIR for relative paths]", base_dir)
    print("[ROOT]", root)
    print("[OUT_DIR]", out_dir)

    # init logging
    tb_writer = None
    if args.tb:
        if SummaryWriter is None:
            raise RuntimeError("TensorBoard not available. Install: pip install tensorboard")
        tb_dir = out_dir / args.tb_dirname
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        print("[TB] logging to:", tb_dir)
        print(f"[TB] run: tensorboard --logdir '{tb_dir}'")

    if args.wandb:
        if wandb is None:
            raise RuntimeError("wandb not installed. Install: pip install wandb")
        run_name = args.wandb_run_name.strip() or f"gru16_bs{args.batch}_lr{args.lr}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        print("[W&B] enabled:", args.wandb_project, "/", run_name)

    # datasets / loaders
    ds_train = SequenceIndexDataset(train_csv, base_dir=base_dir, seq_len=args.seq_len, image_size=args.image_size, strict=args.strict)
    ds_val = SequenceIndexDataset(val_csv, base_dir=base_dir, seq_len=args.seq_len, image_size=args.image_size, strict=args.strict)
    print("[DATA] train:", len(ds_train), "val:", len(ds_val))

    pin = (device.type == "cuda")

    # num_workers 세팅이 크면 hang/느려질 수 있어서 안전 옵션 몇 개 추가
    # (특히 외장 SSD + 많은 workers에서 유용)
    persistent = (args.num_workers > 0)

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # model
    model = ResNet18GRU(
        hidden_size=args.hidden,
        num_layers=args.gru_layers,
        num_classes=len(LABELS),
        pretrained=(not args.no_pretrained),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = None
    if args.amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    best_macro_f1 = -1.0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    start_epoch = 1

    # ✅ resume logic
    if args.resume.strip():
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_absolute():
            resume_path = (out_dir / resume_path).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume not found: {resume_path}")

        print("[RESUME] loading:", resume_path)
        start_epoch, best_macro_f1, ckpt = load_ckpt(resume_path, model, optimizer, scaler, device)

        # optional arg consistency check
        if args.resume_strict_args:
            ck_args = ckpt.get("args", {}) or {}
            def _eq(k):
                return str(ck_args.get(k)) == str(getattr(args, k))
            critical = ["seq_len", "image_size", "hidden", "gru_layers"]
            bad = [k for k in critical if not _eq(k)]
            if bad:
                raise RuntimeError(f"[RESUME] args mismatch for {bad}.\n"
                                   f"ckpt args={ {k: ck_args.get(k) for k in bad} }\n"
                                   f"now args ={ {k: getattr(args,k) for k in bad} }")

        print(f"[RESUME] start_epoch={start_epoch}  best_macro_f1={best_macro_f1:.3f}")

    # training loop
    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_cm, tr_m = run_one_epoch(
            model, dl_train, optimizer, device, scaler, train=True, log_every=args.log_every
        )
        va_loss, va_cm, va_m = run_one_epoch(
            model, dl_val, optimizer=None, device=device, scaler=None, train=False, log_every=args.log_every
        )

        print(f"\n[EPOCH {epoch}/{args.epochs}]")
        print(f"  train loss={tr_loss:.4f}  acc={tr_m['acc']:.3f}  macro_f1={tr_m['macro_f1']:.3f}")
        print(f"  val   loss={va_loss:.4f}  acc={va_m['acc']:.3f}  macro_f1={va_m['macro_f1']:.3f}")
        for i, name in enumerate(LABELS):
            print(f"    - {name:12s}  P={va_m['precision'][i]:.3f} R={va_m['recall'][i]:.3f} "
                  f"F1={va_m['f1'][i]:.3f}  FP={int(va_m['fp'][i])} FN={int(va_m['fn'][i])}")

        # TensorBoard logs (epoch-level)
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", tr_loss, epoch)
            tb_writer.add_scalar("train/acc", float(tr_m["acc"]), epoch)
            tb_writer.add_scalar("train/macro_f1", float(tr_m["macro_f1"]), epoch)

            tb_writer.add_scalar("val/loss", va_loss, epoch)
            tb_writer.add_scalar("val/acc", float(va_m["acc"]), epoch)
            tb_writer.add_scalar("val/macro_f1", float(va_m["macro_f1"]), epoch)

            for i, name in enumerate(LABELS):
                tb_writer.add_scalar(f"val/{name}_precision", float(va_m["precision"][i]), epoch)
                tb_writer.add_scalar(f"val/{name}_recall", float(va_m["recall"][i]), epoch)
                tb_writer.add_scalar(f"val/{name}_f1", float(va_m["f1"][i]), epoch)

            cm_img = cm_to_rgb_image(va_cm)                # HWC uint8
            cm_chw = np.transpose(cm_img, (2, 0, 1))        # CHW
            tb_writer.add_image("val/confusion_matrix", cm_chw, epoch)

        # W&B logs
        if args.wandb and wandb is not None:
            log_dict = {
                "epoch": epoch,
                "train/loss": tr_loss,
                "train/acc": float(tr_m["acc"]),
                "train/macro_f1": float(tr_m["macro_f1"]),
                "val/loss": va_loss,
                "val/acc": float(va_m["acc"]),
                "val/macro_f1": float(va_m["macro_f1"]),
            }
            for i, name in enumerate(LABELS):
                log_dict[f"val/{name}_precision"] = float(va_m["precision"][i])
                log_dict[f"val/{name}_recall"] = float(va_m["recall"][i])
                log_dict[f"val/{name}_f1"] = float(va_m["f1"][i])
            log_dict["val/confusion_matrix"] = wandb.Image(cm_to_rgb_image(va_cm))
            wandb.log(log_dict)

        # save ckpt (last)
        save_ckpt(
            last_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            best_macro_f1=best_macro_f1,
            args=vars(args),
        )

        # save best (by val macro_f1)
        if float(va_m["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(va_m["macro_f1"])
            save_ckpt(
                best_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                best_macro_f1=best_macro_f1,
                args=vars(args),
            )
            print(f"  [SAVE] best -> {best_path} (macro_f1={best_macro_f1:.3f})")

    # close writers
    if tb_writer is not None:
        tb_writer.close()
    if args.wandb and wandb is not None:
        wandb.finish()

    print("\n[DONE]")
    print(" best:", best_path)
    print(" last:", last_path)

    # optional test evaluation (load best)
    if args.eval_test and test_csv.exists():
        print("\n[TEST] evaluating best.pt on test set:", test_csv)
        ds_test = SequenceIndexDataset(test_csv, base_dir=base_dir, seq_len=args.seq_len, image_size=args.image_size, strict=args.strict)
        dl_test = DataLoader(
            ds_test,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
            persistent_workers=persistent,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

        best = torch.load(best_path, map_location=device)
        model.load_state_dict(best["model"])
        model.eval()

        te_loss, te_cm, te_m = run_one_epoch(
            model, dl_test, optimizer=None, device=device, scaler=None, train=False, log_every=args.log_every
        )

        print(f"[TEST] loss={te_loss:.4f} acc={te_m['acc']:.3f} macro_f1={te_m['macro_f1']:.3f}")
        for i, name in enumerate(LABELS):
            print(f"  - {name:12s}  P={te_m['precision'][i]:.3f} R={te_m['recall'][i]:.3f} "
                  f"F1={te_m['f1'][i]:.3f}  FP={int(te_m['fp'][i])} FN={int(te_m['fn'][i])}")

        metrics_path = out_dir / "test_metrics.json"
        out = {
            "test_loss": float(te_loss),
            "test_acc": float(te_m["acc"]),
            "test_macro_f1": float(te_m["macro_f1"]),
            "per_class": {
                LABELS[i]: {
                    "precision": float(te_m["precision"][i]),
                    "recall": float(te_m["recall"][i]),
                    "f1": float(te_m["f1"][i]),
                    "fp": int(te_m["fp"][i]),
                    "fn": int(te_m["fn"][i]),
                } for i in range(len(LABELS))
            },
            "confusion_matrix": te_cm.tolist(),
        }
        metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("[TEST] saved:", metrics_path)


if __name__ == "__main__":
    main()
