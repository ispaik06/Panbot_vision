# resnet18_gru16/tools/make_sequence_index.py
"""
Make sequence index CSVs (train/val/test) from warped frame images and annotations.csv.

Inputs:
- warped frames:
    <Panbot_vision>/resnet18_gru16/data_raw/run_XXXX/warped/img_000001.jpg ...
- annotations:
    <Panbot_vision>/resnet18_gru16/assets/annotations.csv
    columns: run_id,start_frame,end_frame,label
    (헤더/값에 공백 있어도 자동 strip 처리)

Outputs:
- <Panbot_vision>/resnet18_gru16/dataset/index_train.csv
- <Panbot_vision>/resnet18_gru16/dataset/index_val.csv
- <Panbot_vision>/resnet18_gru16/dataset/index_test.csv

Each row corresponds to one sample sequence:
- run_id
- end_frame (the last frame number in the sequence)
- label (derived from end_frame)
- frame_00 ... frame_15
  (PATHS STORED AS RELATIVE PATHS FROM Panbot_vision)

Label rule:
- label is determined by the segment that CONTAINS end_frame.

Split policy (improved):
- 기본은 "run-level split" 이지만,
- 실제 생성되는 sequence 개수(= index row 수)를 기준으로
  train/val/test 비율(기본 0.8/0.1/0.1)을 최대한 맞추고,
  가능하면 val/test에 모든 라벨이 최소 1개 이상 포함되도록 greedy 배치.

Recommended usage:
python resnet18_gru16/tools/make_sequence_index.py --stride 6 --seq_len 16 --margin 200
"""

import argparse
import csv
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FRAME_RE = re.compile(r".*?(\d+)$")  # trailing digits in stem

LABELS = ["not_ready", "almost_ready", "ready"]


@dataclass
class Segment:
    start: int  # inclusive
    end: int    # inclusive
    label: str


def extract_frame_number_from_path(p: Path) -> Optional[int]:
    m = FRAME_RE.match(p.stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def list_frames_sorted(frames_dir: Path) -> List[Tuple[int, Path]]:
    items: List[Tuple[int, Path]] = []
    for p in frames_dir.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith("._"):  # macOS AppleDouble 제거
            continue
        if p.suffix.lower() in IMG_EXTS:
            n = extract_frame_number_from_path(p)
            if n is not None:
                items.append((n, p))
    items.sort(key=lambda x: x[0])
    return items


def read_annotations(csv_path: Path) -> Dict[str, List[Segment]]:
    """
    annotations.csv를 robust하게 읽습니다.
    - utf-8-sig(BOM) 허용
    - 헤더 이름 strip 처리 (예: 'run_id ' -> 'run_id')
    - 각 값도 strip 처리
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"annotations.csv not found: {csv_path}")

    segments_by_run: Dict[str, List[Segment]] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        # ✅ 헤더 공백 처리
        if reader.fieldnames:
            reader.fieldnames = [h.strip() if h is not None else "" for h in reader.fieldnames]

        required = {"run_id", "start_frame", "end_frame", "label"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"annotations.csv must contain columns: {sorted(required)}\n"
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            # ✅ 값 공백 처리
            run_id = (row.get("run_id") or "").strip()
            start_s = (row.get("start_frame") or "").strip()
            end_s = (row.get("end_frame") or "").strip()
            label = (row.get("label") or "").strip()

            if not run_id or not start_s or not end_s or not label:
                continue

            # label normalize (필요 시)
            label = label.strip()
            if label not in LABELS:
                # 라벨이 예상과 다르면 스킵 (원하시면 여기서 예외로 바꿔도 됨)
                continue

            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                raise ValueError(f"Invalid start/end in annotations: {run_id}, {start_s}, {end_s}")

            if end < start:
                raise ValueError(f"Invalid segment (end < start): {run_id} {start}-{end}")

            segments_by_run.setdefault(run_id, []).append(Segment(start=start, end=end, label=label))

    for rid in segments_by_run:
        segments_by_run[rid].sort(key=lambda s: (s.start, s.end))

    return segments_by_run


def label_for_frame(segments: List[Segment], frame_num: int, margin: int = 0) -> Optional[str]:
    """
    Return label for frame_num based on segments.
    margin: exclude +/- margin around segment boundaries.
    """
    if not segments:
        return None

    if margin > 0:
        boundaries = set()
        for s in segments:
            boundaries.add(s.start)
            boundaries.add(s.end)
            boundaries.add(s.end + 1)
        for b in boundaries:
            if abs(frame_num - b) <= margin:
                return None

    for s in segments:
        if s.start <= frame_num <= s.end:
            return s.label
    return None


def write_index_csv(out_path: Path, rows: List[dict], seq_len: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "end_frame", "label"] + [f"frame_{i:02d}" for i in range(seq_len)]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def count_sequences_for_run(
    run_id: str,
    frames_dir: Path,
    segments: List[Segment],
    seq_len: int,
    stride: int,
    margin: int,
) -> Tuple[int, Dict[str, int]]:
    """
    split 품질을 위해, '이 run이 생성하는 sequence 수'와 '라벨별 sequence 수'를 먼저 추정합니다.
    (실제 rows 저장 없이 count만 계산)
    """
    items = list_frames_sorted(frames_dir)
    if not items:
        return 0, {k: 0 for k in LABELS}

    fmap = {n: p for n, p in items}
    frame_nums = [n for n, _ in items]
    min_n = frame_nums[0]
    max_n = frame_nums[-1]

    total = 0
    per_label = {k: 0 for k in LABELS}

    for k in frame_nums:
        req0 = k - stride * (seq_len - 1)
        if req0 < min_n or k > max_n:
            continue

        # all required frames exist?
        ok = True
        for i in range(seq_len):
            n = k - stride * (seq_len - 1 - i)
            if fmap.get(n) is None:
                ok = False
                break
        if not ok:
            continue

        lab = label_for_frame(segments, k, margin=margin)
        if lab is None:
            continue

        total += 1
        if lab in per_label:
            per_label[lab] += 1

    return total, per_label


def build_rows_for_run(
    run_id: str,
    frames_dir: Path,
    segments: List[Segment],
    seq_len: int,
    stride: int,
    margin: int,
    panbot_root: Path,  # Panbot_vision absolute path
) -> List[dict]:
    """
    Build actual sequence rows for one run.
    Stored paths are RELATIVE to panbot_root (Panbot_vision).
    """
    items = list_frames_sorted(frames_dir)
    if not items:
        return []

    fmap = {n: p for n, p in items}
    frame_nums = [n for n, _ in items]
    min_n = frame_nums[0]
    max_n = frame_nums[-1]

    rows: List[dict] = []

    for k in frame_nums:
        req = [k - stride * (seq_len - 1 - i) for i in range(seq_len)]
        if req[0] < min_n or req[-1] > max_n:
            continue

        paths: List[Path] = []
        ok = True
        for n in req:
            p = fmap.get(n)
            if p is None:
                ok = False
                break
            paths.append(p)
        if not ok:
            continue

        lab = label_for_frame(segments, k, margin=margin)
        if lab is None:
            continue

        row = {"run_id": run_id, "end_frame": k, "label": lab}
        for i, p in enumerate(paths):
            rel = os.path.relpath(str(p.resolve()), start=str(panbot_root))
            row[f"frame_{i:02d}"] = Path(rel).as_posix()
        rows.append(row)

    return rows


def greedy_stratified_split_by_sequence(
    run_infos: Dict[str, Dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    ensure_label_coverage: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """
    run_infos[run_id] = {
      "total": int,
      "per_label": {label:int}
    }

    목표: 전체 sequence 수 기준 train/val/test 비율을 맞추고,
         가능하면 val/test에도 모든 라벨이 1개 이상 포함되게 greedy 배치.
    """
    rng = random.Random(seed)

    run_ids = list(run_infos.keys())
    # 큰 run부터 배치하는 게 ratio 맞추기 쉽습니다
    run_ids.sort(key=lambda rid: run_infos[rid]["total"], reverse=True)

    total_seq = sum(run_infos[r]["total"] for r in run_ids)
    target_train = total_seq * train_ratio
    target_val = total_seq * val_ratio
    target_test = total_seq - target_train - target_val

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_label_counts = {
        "train": {k: 0 for k in LABELS},
        "val": {k: 0 for k in LABELS},
        "test": {k: 0 for k in LABELS},
    }
    targets = {"train": target_train, "val": target_val, "test": target_test}

    # 작은 랜덤 타이브레이커
    jitter = {rid: rng.random() * 1e-6 for rid in run_ids}

    def missing_labels(split_name: str) -> List[str]:
        return [k for k in LABELS if split_label_counts[split_name][k] == 0]

    def score_for(split_name: str, rid: str) -> float:
        run_total = run_infos[rid]["total"]
        run_per = run_infos[rid]["per_label"]

        # 1) ratio 기반: 목표 대비 오차
        after = split_counts[split_name] + run_total
        base = abs(after - targets[split_name]) / max(targets[split_name], 1.0)

        # 2) 라벨 커버리지 보상 (val/test에 특히 강하게)
        reward = 0
        if ensure_label_coverage:
            miss = missing_labels(split_name)
            for lab in miss:
                if run_per.get(lab, 0) > 0:
                    reward += 1

        # val/test는 라벨 커버리지를 더 중요시
        if split_name in ("val", "test"):
            base -= 0.35 * reward
        else:
            base -= 0.10 * reward

        # 3) 너무 초과하면 패널티
        overflow = max(0.0, after - targets[split_name])
        base += (overflow / max(targets[split_name], 1.0)) * 0.25

        return base + jitter[rid]

    # 1차 배치
    for rid in run_ids:
        # 시퀀스가 0이면 의미 없으니 train에 넣고 넘어감(혹은 완전 제외)
        if run_infos[rid]["total"] <= 0:
            splits["train"].append(rid)
            continue

        # 점수 계산 후 최적 split 선택
        choices = ["train", "val", "test"]
        best = min(choices, key=lambda s: score_for(s, rid))
        splits[best].append(rid)
        split_counts[best] += run_infos[rid]["total"]
        for lab in LABELS:
            split_label_counts[best][lab] += run_infos[rid]["per_label"].get(lab, 0)

    # 2차 보정: val/test 라벨 누락이면 train에서 가져와 보정
    if ensure_label_coverage:
        for target_split in ("val", "test"):
            miss = missing_labels(target_split)
            if not miss:
                continue

            # 누락 라벨마다 하나씩 채우기 시도
            for lab in miss:
                # train에서 해당 라벨을 가진 run 후보 찾기
                candidates = [rid for rid in splits["train"] if run_infos[rid]["per_label"].get(lab, 0) > 0]
                if not candidates:
                    continue

                # "가져왔을 때" ratio가 덜 깨지는 후보를 선택
                def move_cost(rid: str) -> float:
                    rt = run_infos[rid]["total"]
                    # train에서 빼고 target_split에 더했을 때 목표 오차 합
                    new_train = split_counts["train"] - rt
                    new_tgt = split_counts[target_split] + rt
                    c = abs(new_train - targets["train"]) / max(targets["train"], 1.0)
                    c += abs(new_tgt - targets[target_split]) / max(targets[target_split], 1.0)
                    # target_split에서 새로 커버되는 라벨이면 보상
                    if split_label_counts[target_split][lab] == 0:
                        c -= 0.2
                    return c

                best_rid = min(candidates, key=move_cost)

                # 이동 실행
                splits["train"].remove(best_rid)
                splits[target_split].append(best_rid)

                rt = run_infos[best_rid]["total"]
                split_counts["train"] -= rt
                split_counts[target_split] += rt
                for l2 in LABELS:
                    split_label_counts["train"][l2] -= run_infos[best_rid]["per_label"].get(l2, 0)
                    split_label_counts[target_split][l2] += run_infos[best_rid]["per_label"].get(l2, 0)

    return splits["train"], splits["val"], splits["test"]


def main():
    ap = argparse.ArgumentParser(description="Create train/val/test index CSVs for ResNet18+GRU16 dataset.")
    ap.add_argument("--root", type=str, default=None, help="resnet18_gru16 root (default: auto-detect).")
    ap.add_argument("--data_raw", type=str, default="data_raw", help="data_raw dir under root (default: data_raw)")
    ap.add_argument("--in_subdir", type=str, default="warped", help="Input subdir inside each run (default: warped)")

    ap.add_argument("--annotations", type=str, default="assets/annotations.csv", help="annotations.csv under root.")
    ap.add_argument("--out_dir", type=str, default="dataset", help="Output dir under root (default: dataset)")

    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--margin", type=int, default=0)

    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--only_runs", type=str, default="",
                    help="Optional: comma-separated run_ids to include (e.g., run_0001,run_0002). Empty=all in annotations.")

    # ✅ split behavior
    ap.add_argument("--split_mode", type=str, default="seq",
                    choices=["seq", "run"],
                    help="seq: split by sequence counts (recommended). run: split by run counts (old).")
    ap.add_argument("--no_label_coverage", action="store_true",
                    help="Disable label coverage enforcement for val/test.")

    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    default_root = script_path.parent.parent  # .../resnet18_gru16
    root = Path(args.root).resolve() if args.root else default_root

    # ✅ Panbot_vision root is parent of resnet18_gru16
    panbot_root = root.parent.resolve()

    data_raw = (root / args.data_raw).resolve()
    if not data_raw.exists():
        raise FileNotFoundError(f"data_raw not found: {data_raw}")

    ann_path = (root / args.annotations).resolve()
    segments_by_run = read_annotations(ann_path)

    if args.only_runs.strip():
        allowed = [x.strip() for x in args.only_runs.split(",") if x.strip()]
        allowed_set = set(allowed)
        segments_by_run = {k: v for k, v in segments_by_run.items() if k in allowed_set}

    run_ids = sorted(list(segments_by_run.keys()))
    if not run_ids:
        print("[INFO] No runs found in annotations.")
        return

    print("[ROOT] resnet18_gru16:", root)
    print("[ROOT] Panbot_vision :", panbot_root)

    # ✅ split 위해 run별 sequence 수를 먼저 계산
    run_infos: Dict[str, Dict] = {}
    for rid in run_ids:
        frames_dir = data_raw / rid / args.in_subdir
        if not frames_dir.exists():
            run_infos[rid] = {"total": 0, "per_label": {k: 0 for k in LABELS}}
            continue
        total, per = count_sequences_for_run(
            run_id=rid,
            frames_dir=frames_dir,
            segments=segments_by_run.get(rid, []),
            seq_len=args.seq_len,
            stride=args.stride,
            margin=args.margin,
        )
        run_infos[rid] = {"total": total, "per_label": per}

    # Split
    ensure_cov = (not args.no_label_coverage)

    if args.split_mode == "seq":
        train_runs, val_runs, test_runs = greedy_stratified_split_by_sequence(
            run_infos=run_infos,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            ensure_label_coverage=ensure_cov,
        )
        split_note = "greedy-stratified (sequence-count based)"
    else:
        # old run-count based split
        rng = random.Random(args.seed)
        rids = list(run_ids)
        rng.shuffle(rids)
        n = len(rids)
        n_train = int(round(n * args.train_ratio))
        n_val = int(round(n * args.val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train_runs = rids[:n_train]
        val_runs = rids[n_train:n_train + n_val]
        test_runs = rids[n_train + n_val:]
        split_note = "simple shuffle (run-count based)"

    def seq_sum(runs: List[str]) -> int:
        return sum(run_infos[r]["total"] for r in runs)

    print("[SPLIT_MODE]", args.split_mode, f"({split_note})")
    print("[SPLIT] runs count:", len(run_ids))
    print("  - train:", len(train_runs), train_runs[:10], ("..." if len(train_runs) > 10 else ""))
    print("  - val  :", len(val_runs), val_runs[:10], ("..." if len(val_runs) > 10 else ""))
    print("  - test :", len(test_runs), test_runs[:10], ("..." if len(test_runs) > 10 else ""))

    total_seq = sum(run_infos[r]["total"] for r in run_ids)
    tr_seq, va_seq, te_seq = seq_sum(train_runs), seq_sum(val_runs), seq_sum(test_runs)
    print(f"[SPLIT_SEQ] total={total_seq} | train={tr_seq} ({tr_seq/max(total_seq,1)*100:.1f}%) "
          f"val={va_seq} ({va_seq/max(total_seq,1)*100:.1f}%) test={te_seq} ({te_seq/max(total_seq,1)*100:.1f}%)")

    def label_cov(runs: List[str]) -> Dict[str, int]:
        cov = {k: 0 for k in LABELS}
        for r in runs:
            for k in LABELS:
                cov[k] += run_infos[r]["per_label"].get(k, 0)
        return cov

    cov_tr = label_cov(train_runs)
    cov_va = label_cov(val_runs)
    cov_te = label_cov(test_runs)
    print("[LABEL_COV] train:", cov_tr)
    print("[LABEL_COV] val  :", cov_va)
    print("[LABEL_COV] test :", cov_te)

    # Build rows and write CSVs
    def build_for_runs(runs: List[str]) -> List[dict]:
        all_rows: List[dict] = []
        for run_id in runs:
            run_dir = data_raw / run_id
            frames_dir = run_dir / args.in_subdir
            segs = segments_by_run.get(run_id, [])
            if not frames_dir.exists():
                print(f"[SKIP] {run_id}: missing frames dir: {frames_dir}")
                continue

            rows = build_rows_for_run(
                run_id=run_id,
                frames_dir=frames_dir,
                segments=segs,
                seq_len=args.seq_len,
                stride=args.stride,
                margin=args.margin,
                panbot_root=panbot_root,
            )
            print(f"[RUN] {run_id}: sequences={len(rows)}")
            all_rows.extend(rows)
        return all_rows

    train_rows = build_for_runs(train_runs)
    val_rows = build_for_runs(val_runs)
    test_rows = build_for_runs(test_runs)

    out_dir = (root / args.out_dir).resolve()
    out_train = out_dir / "index_train.csv"
    out_val = out_dir / "index_val.csv"
    out_test = out_dir / "index_test.csv"

    write_index_csv(out_train, train_rows, args.seq_len)
    write_index_csv(out_val, val_rows, args.seq_len)
    write_index_csv(out_test, test_rows, args.seq_len)

    print("[DONE]")
    print("  -", out_train)
    print("  -", out_val)
    print("  -", out_test)
    print("[NOTE] frame_* paths are stored RELATIVE to Panbot_vision.")
    print("[NOTE] split was", split_note)


if __name__ == "__main__":
    main()
