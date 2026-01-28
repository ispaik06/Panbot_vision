# tools/rename_dataset.py
import argparse
import sys
import uuid
from pathlib import Path

EXTS_DEFAULT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_by_stem(folder: Path, exts: set[str]):
    """
    folder 내 파일을 stem 기준으로 모아서 dict[stem] = [paths...] 형태로 반환
    """
    d: dict[str, list[Path]] = {}
    if not folder.exists():
        return d
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        d.setdefault(p.stem, []).append(p)
    return d


def pick_single_file(stem: str, paths: list[Path], kind: str):
    """
    한 stem에 파일이 2개 이상이면 에러로 처리.
    """
    if len(paths) == 1:
        return paths[0]
    raise RuntimeError(
        f"[{kind}] 같은 이름(stem) '{stem}'에 파일이 {len(paths)}개 있습니다: "
        + ", ".join(p.name for p in paths)
    )


def build_matches(dir1: Path, dir2: Path, exts1: set[str], exts2: set[str]):
    """
    매칭 규칙: stem(확장자 제외 파일명)이 완전히 같은 것만 매칭
    """
    m1_raw = collect_by_stem(dir1, exts1)
    m2_raw = collect_by_stem(dir2, exts2)

    # stem 당 파일 1개로 확정(중복 stem은 에러)
    m1: dict[str, Path] = {}
    for stem, paths in m1_raw.items():
        m1[stem] = pick_single_file(stem, paths, "DIR1")
    m2: dict[str, Path] = {}
    for stem, paths in m2_raw.items():
        m2[stem] = pick_single_file(stem, paths, "DIR2")

    stems1 = set(m1.keys())
    stems2 = set(m2.keys())

    common = sorted(stems1 & stems2)
    matches = [(stem, m1[stem], m2[stem]) for stem in common]

    only_dir1 = sorted([m1[s] for s in (stems1 - stems2)], key=lambda p: p.name)
    only_dir2 = sorted([m2[s] for s in (stems2 - stems1)], key=lambda p: p.name)

    return matches, only_dir1, only_dir2


def make_plan(matches_sorted, dir1: Path, dir2: Path, prefix: str, width: int):
    """
    리네임 계획: (src -> dst) 목록 생성
    """
    plan: list[tuple[Path, Path]] = []
    used_targets = set()

    for i, (_stem, p1, p2) in enumerate(matches_sorted, start=1):
        new_stem = f"{prefix}{i:0{width}d}"

        dst1 = dir1 / f"{new_stem}{p1.suffix.lower()}"
        dst2 = dir2 / f"{new_stem}{p2.suffix.lower()}"

        if dst1 in used_targets or dst2 in used_targets:
            raise RuntimeError(f"중복 타겟 발생: {dst1} 또는 {dst2}")
        used_targets.add(dst1)
        used_targets.add(dst2)

        plan.append((p1, dst1))
        plan.append((p2, dst2))

    return plan


def print_mismatch_report(only_dir1, only_dir2, max_show=30):
    if not only_dir1 and not only_dir2:
        return False

    print("\n[WARN] 두 폴더 간 매칭되지 않은 파일이 있습니다.\n", file=sys.stderr)

    if only_dir1:
        print(f"- DIR1에만 존재: {len(only_dir1)}개", file=sys.stderr)
        for p in only_dir1[:max_show]:
            print(f"  * {p.name}", file=sys.stderr)
        if len(only_dir1) > max_show:
            print(f"  ... (+{len(only_dir1)-max_show} more)", file=sys.stderr)

    if only_dir2:
        print(f"\n- DIR2에만 존재: {len(only_dir2)}개", file=sys.stderr)
        for p in only_dir2[:max_show]:
            print(f"  * {p.name}", file=sys.stderr)
        if len(only_dir2) > max_show:
            print(f"  ... (+{len(only_dir2)-max_show} more)", file=sys.stderr)

    print("", file=sys.stderr)
    return True


def apply_plan(plan, dry_run: bool):
    for src, dst in plan:
        print(f"{src.name}  ->  {dst.name}")

    if dry_run:
        print("\n(dry-run) 실제 변경은 하지 않았습니다.")
        return

    # 충돌 방지를 위한 2단계 리네임
    tmp_map: list[tuple[Path, Path]] = []
    for src, dst in plan:
        if not src.exists():
            raise FileNotFoundError(f"원본 파일 없음: {src}")
        tmp = src.with_name(f".__tmp__{uuid.uuid4().hex}__{src.name}")
        src.rename(tmp)
        tmp_map.append((tmp, dst))

    for tmp, dst in tmp_map:
        if dst.exists():
            raise FileExistsError(f"이미 존재하는 타겟 파일: {dst}")
        tmp.rename(dst)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dir1", type=str, required=True, help="첫 번째 폴더 경로")
    ap.add_argument("--dir2", type=str, required=True, help="두 번째 폴더 경로")

    ap.add_argument("--exts1", type=str, default="", help="DIR1 허용 확장자 콤마구분 (예: .jpg,.png). 비우면 기본 이미지 확장자")
    ap.add_argument("--exts2", type=str, default="", help="DIR2 허용 확장자 콤마구분 (예: .png). 비우면 기본 이미지 확장자")

    ap.add_argument("--prefix", type=str, default="item_", help="새 파일명 prefix")
    ap.add_argument("--width", type=int, default=6, help="숫자 자리수 (기본 6: 000001)")
    ap.add_argument("--apply", action="store_true", help="실제 적용 (없으면 dry-run)")
    ap.add_argument("--abort_on_mismatch", action="store_true",
                    help="매칭되지 않은 파일이 하나라도 있으면 리네임을 중단")

    args = ap.parse_args()

    dir1 = Path(args.dir1).expanduser().resolve()
    dir2 = Path(args.dir2).expanduser().resolve()

    if not dir1.exists():
        raise FileNotFoundError(f"DIR1 폴더가 없습니다: {dir1}")
    if not dir2.exists():
        raise FileNotFoundError(f"DIR2 폴더가 없습니다: {dir2}")

    def parse_exts(s: str):
        if not s.strip():
            return set(EXTS_DEFAULT)
        out = set()
        for tok in s.split(","):
            tok = tok.strip().lower()
            if not tok:
                continue
            if not tok.startswith("."):
                tok = "." + tok
            out.add(tok)
        return out

    exts1 = parse_exts(args.exts1)
    exts2 = parse_exts(args.exts2)

    matches, only_dir1, only_dir2 = build_matches(dir1, dir2, exts1, exts2)
    has_mismatch = print_mismatch_report(only_dir1, only_dir2)

    if args.abort_on_mismatch and has_mismatch:
        print("[ABORT] 매칭되지 않은 파일이 있어 작업을 중단합니다. (--abort_on_mismatch)", file=sys.stderr)
        sys.exit(2)

    if not matches:
        print("매칭되는 항목이 없습니다. 작업 종료.")
        return

    # stem 이름 기준 정렬(안전/예측가능)
    matches_sorted = sorted(matches, key=lambda x: x[0])

    plan = make_plan(matches_sorted, dir1, dir2, args.prefix, args.width)

    print(f"총 매칭 항목: {len(matches_sorted)} (총 {len(plan)}개 파일 리네임)")
    apply_plan(plan, dry_run=(not args.apply))


if __name__ == "__main__":
    main()
