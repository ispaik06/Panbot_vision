import argparse
from pathlib import Path

from huggingface_hub import create_repo, snapshot_download, upload_file
from ultralytics import YOLO


def resolve_project_root() -> Path:
    # .../PANBOT_VISION/yolov8/scripts/train_seg.py -> parents[2] = PANBOT_VISION
    return Path(__file__).resolve().parents[2]


def resolve_path(project_root: Path, p: str) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def download_hf_dataset_yaml(hf_dataset_repo: str, token: str | None) -> Path:
    """
    HF dataset repo를 로컬로 다운로드하고 dataset.yaml 경로를 반환합니다.

    업로드한 repo 구조 가정:
      dataset.yaml
      yolo/images/train/...
      yolo/images/val/...
      yolo/labels/train/...
      yolo/labels/val/...
    """
    local_dir = snapshot_download(
        repo_id=hf_dataset_repo,
        repo_type="dataset",
        token=token,
    )
    local_root = Path(local_dir).resolve()
    yaml_path = local_root / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"[ERROR] dataset.yaml not found in downloaded HF dataset repo: {yaml_path}"
        )
    return yaml_path


def upload_best_to_hf_model(best_pt: Path, hf_model_repo: str, token: str | None):
    """
    best.pt + README 템플릿 파일을 HF model repo(public)에 업로드합니다.
    README 내용은 코드에 박지 않고, yolov8/assets/README_model.md 파일을 그대로 올립니다.
    """
    project_root = resolve_project_root()

    # public model repo 생성(없으면)
    create_repo(
        repo_id=hf_model_repo,
        repo_type="model",
        private=False,
        exist_ok=True,
        token=token,
    )

    # best.pt 업로드
    upload_file(
        repo_id=hf_model_repo,
        repo_type="model",
        path_or_fileobj=str(best_pt),
        path_in_repo="best.pt",
        token=token,
    )

    # README 템플릿 업로드
    readme_src = project_root / "yolov8" / "assets" / "README_model.md"
    if not readme_src.exists():
        raise FileNotFoundError(
            f"[ERROR] README template not found: {readme_src}\n"
            f"Create it first: PANBOT_VISION/yolov8/assets/README_model.md"
        )

    upload_file(
        repo_id=hf_model_repo,
        repo_type="model",
        path_or_fileobj=str(readme_src),
        path_in_repo="README.md",
        token=token,
    )

    print("[OK] Uploaded to HF model repo (public):")
    print(f" - repo_id: {hf_model_repo}")
    print(" - files : best.pt, README.md")


def main():
    ap = argparse.ArgumentParser()

    # dataset input
    ap.add_argument(
        "--dataset_yaml",
        type=str,
        default="",
        help="local dataset.yaml path. If empty, use --hf_dataset_repo.",
    )
    ap.add_argument(
        "--hf_dataset_repo",
        type=str,
        default="",
        help="HF dataset repo_id, e.g. username/panbot_batter_yolo",
    )

    # training options
    ap.add_argument(
        "--pretrained",
        type=str,
        default="yolov8n-seg.pt",
        help="pretrained base model (yolov8n-seg.pt / yolov8s-seg.pt ...)",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--device",
        type=str,
        default="",
        help="''=auto, 'cpu', '0', '0,1' etc.",
    )
    ap.add_argument(
        "--name",
        type=str,
        default="batter_seg_v1",
        help="run name (runs/.../{name})",
    )
    ap.add_argument(
        "--project",
        type=str,
        default="yolov8/runs",
        help="runs output root (relative to PANBOT_VISION if not absolute)",
    )

    # optional HF model upload
    ap.add_argument(
        "--hf_model_repo",
        type=str,
        default="",
        help="HF model repo_id to upload best.pt (public). optional.",
    )
    ap.add_argument(
        "--token",
        type=str,
        default="",
        help="HF token. optional (if empty uses cached login).",
    )

    args = ap.parse_args()

    token = args.token if args.token else None
    project_root = resolve_project_root()

    # resolve dataset.yaml
    if args.dataset_yaml:
        dataset_yaml = resolve_path(project_root, args.dataset_yaml)
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"[ERROR] local dataset.yaml not found: {dataset_yaml}")
    else:
        if not args.hf_dataset_repo:
            raise ValueError("[ERROR] Provide --dataset_yaml OR --hf_dataset_repo")
        dataset_yaml = download_hf_dataset_yaml(args.hf_dataset_repo, token)

    # resolve runs project
    runs_project = resolve_path(project_root, args.project)
    runs_project.mkdir(parents=True, exist_ok=True)

    print("[INFO] project_root :", project_root)
    print("[INFO] dataset_yaml :", dataset_yaml)
    print("[INFO] runs_project :", runs_project)
    print("[INFO] pretrained   :", args.pretrained)

    # train
    model = YOLO(args.pretrained)

    train_kwargs = dict(
        task="segment",
        data=str(dataset_yaml),
        imgsz=int(args.imgsz),
        epochs=int(args.epochs),
        batch=int(args.batch),
        name=str(args.name),
        project=str(runs_project),
    )
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"[ERROR] best.pt not found: {best_pt}")

    print("[DONE] Training finished.")
    print(" - save_dir:", save_dir)
    print(" - best.pt :", best_pt)

    # upload best.pt to HF model repo if requested
    if args.hf_model_repo:
        upload_best_to_hf_model(best_pt, args.hf_model_repo, token)


if __name__ == "__main__":
    main()