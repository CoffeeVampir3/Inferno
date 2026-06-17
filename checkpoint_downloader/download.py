import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "google/gemma-4-26B-A4B"
DEFAULT_CHECKPOINTS = REPO_ROOT / "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model snapshot into checkpoints/<name>."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--name", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--allow", nargs="*", default=None)
    parser.add_argument("--ignore", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.name or args.repo.rstrip("/").split("/")[-1]
    target = args.checkpoints / name
    target.mkdir(parents=True, exist_ok=True)

    print(f"downloading {args.repo} -> {target}")
    path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=target,
        token=args.token,
        allow_patterns=args.allow,
        ignore_patterns=args.ignore,
    )
    print(f"done: {path}")


if __name__ == "__main__":
    main()
