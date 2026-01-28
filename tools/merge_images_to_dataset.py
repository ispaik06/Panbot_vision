# tools/merge_images_to_dataset.py
import argparse
import re
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_max_index(dst_dir: Path, prefix: str = "img", width: int = 6):
    """
    dst_dir 안의 prefix_000001.* 형태에서 가장 큰 번호를 찾아 반환.
    없으면 0.
    """
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    best = 0
    for p in dst_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        m = pat.match(p.stem)
        if not m:
            continue
        try:
            best = max(best, int(m.group(1)))
        except Exception:
            pass
    return best

def list_images_recursive(folder: Path, recursive: bool):
    if recursive:
        it = folder.rglob("*")
    else:
        it = folder.iterdir()

    files = []
    for p in it:
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    files.sort(key=lambda p: p.name)  # 각 폴더 내부 순서(예측가능)
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", type=str, default="images", help="대상 폴더 (기본: Panbot_vision/images)")
    ap.add_argument("--src", type=str, nargs="+", required=True,
                    help="복사할 소스 폴더들 (여러 개 가능). 예: raw_datasets/warped_images/20260125 raw_datasets/warped_images/20260126")
    ap.add_argument("--prefix", type=str, default="img", help="출력 파일 prefix (기본 img)")
    ap.add_argument("--width", type=int, default=6, help="번호 자리수 (기본 6 -> 000001)")
    ap.add_argument("--recursive", action="store_true", help="소스 폴더를 재귀적으로 탐색")
    ap.add_argument("--ext", type=str, default="", help="출력 확장자 고정(.jpg/.png). 비우면 원본 확장자 유지")
    ap.add_argument("--dry_run", action="store_true", help="실제 복사 없이 계획만 출력")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    dst_dir = Path(args.dst)
    if not dst_dir.is_absolute():
        dst_dir = (project_root / dst_dir).resolve()

    ensure_dir(dst_dir)

    fixed_ext = args.ext.strip().lower()
    if fixed_ext and not fixed_ext.startswith("."):
        fixed_ext = "." + fixed_ext
    if fixed_ext and fixed_ext not in IMG_EXTS:
        raise ValueError(f"--ext는 이미지 확장자여야 합니다: {fixed_ext}")

    max_idx = find_max_index(dst_dir, prefix=args.prefix, width=args.width)
    next_idx = max_idx + 1

    print(f"[DST] {dst_dir}")
    print(f"[DST] existing max index = {max_idx} -> start from {next_idx:0{args.width}d}")
    print(f"[OPT] recursive={args.recursive}, fixed_ext={fixed_ext or '(keep original)'}")
    if args.dry_run:
        print("[MODE] dry-run (복사하지 않음)")

    copied = 0
    for src in args.src:
        src_dir = Path(src)
        if not src_dir.is_absolute():
            src_dir = (project_root / src_dir).resolve()
        if not src_dir.exists():
            raise FileNotFoundError(f"소스 폴더가 없습니다: {src_dir}")

        files = list_images_recursive(src_dir, recursive=args.recursive)
        print(f"\n[SRC] {src_dir}  (files={len(files)})")

        for p in files:
            out_ext = fixed_ext if fixed_ext else p.suffix.lower()
            dst_name = f"{args.prefix}_{next_idx:0{args.width}d}{out_ext}"
            dst_path = dst_dir / dst_name

            # 안전장치: 혹시라도 존재하면 다음 번호로 밀기
            while dst_path.exists():
                next_idx += 1
                dst_name = f"{args.prefix}_{next_idx:0{args.width}d}{out_ext}"
                dst_path = dst_dir / dst_name

            print(f"{p}  ->  {dst_path.name}")

            if not args.dry_run:
                shutil.copy2(p, dst_path)

            copied += 1
            next_idx += 1

    print(f"\n[DONE] copied={copied}, last_index={(next_idx-1):0{args.width}d}")

if __name__ == "__main__":
    main()
