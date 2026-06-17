import argparse
import csv
import random
from pathlib import Path

csv.field_size_limit(10 ** 7)

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
LEVELS = ["high", "low"]

HF_REPO_ID = "wenkai-li/big5_chat"


def ensure_csv(csv_path, repo_id, allow_download):
    path = Path(csv_path)
    if path.exists():
        return path
    if not allow_download:
        raise SystemExit(
            f"source CSV not found: {path}\n"
            f"drop --no-download to fetch it from the '{repo_id}' HF dataset.")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "source CSV missing and huggingface_hub is not installed.\n"
            f"run `pixi run pip install huggingface_hub`, or place the file at {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"source CSV not found; downloading {path.name} from {repo_id} ...")
    local = hf_hub_download(
        repo_id=repo_id, filename=path.name, repo_type="dataset",
        local_dir=str(path.parent))
    print(f"downloaded to {local}")
    return Path(local)


def sanitize(text):
    return " ".join(text.replace("\r", " ").replace("\n", " ").replace("\t", " ").split())


def load_pool(csv_path):
    pool = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trait = row.get("trait", "")
            level = row.get("level", "")
            inp = sanitize(row.get("train_input", "") or "")
            out = sanitize(row.get("train_output", "") or "")
            if not inp or not out:
                continue
            pool.setdefault((trait, level), []).append((inp, out))
    return pool


def dedupe(pairs):
    seen = set()
    result = []
    for inp, out in pairs:
        if out in seen:
            continue
        seen.add(out)
        result.append((inp, out))
    return result


def write_split(out_dir, trait, level, split, pairs):
    in_path = out_dir / f"{trait}_{level}_{split}_in.txt"
    out_path = out_dir / f"{trait}_{level}_{split}_out.txt"
    with open(in_path, "w") as fi, open(out_path, "w") as fo:
        for inp, out in pairs:
            fi.write(inp + "\n")
            fo.write(out + "\n")
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(
        description="Re-curate Big Five steering data from the source pool.")
    parser.add_argument("--csv", default="steering_data/big5_chat_dataset.csv",
                        help="source CSV path")
    parser.add_argument("--out", default="steering_data_big",
                        help="output directory")
    parser.add_argument("--train", type=int, default=2000,
                        help="train examples per class")
    parser.add_argument("--eval", type=int, default=256,
                        help="eval examples per class (disjoint from train)")
    parser.add_argument("--seed", type=int, default=1337,
                        help="shuffle seed for reproducible curation")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="keep duplicate outputs")
    parser.add_argument("--traits", nargs="+", default=TRAITS,
                        help="subset of traits to curate")
    parser.add_argument("--repo-id", default=HF_REPO_ID,
                        help="HF dataset repo to fetch the source CSV from when missing")
    parser.add_argument("--no-download", action="store_true",
                        help="never fetch; require the source CSV to already exist")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = ensure_csv(args.csv, args.repo_id, allow_download=not args.no_download)
    pool = load_pool(csv_path)
    rng = random.Random(args.seed)

    print(f"output: {out_dir}  | train/class {args.train}  | eval/class {args.eval}")
    for trait in args.traits:
        for level in LEVELS:
            pairs = pool.get((trait, level), [])
            if not args.no_dedupe:
                pairs = dedupe(pairs)
            rng.shuffle(pairs)
            need = args.train + args.eval
            chosen = pairs[:need]
            train_pairs = chosen[:args.train]
            eval_pairs = chosen[args.train:args.train + args.eval]
            nt = write_split(out_dir, trait, level, "train", train_pairs)
            ne = write_split(out_dir, trait, level, "eval", eval_pairs)
            print(f"  {trait:18s} {level:4s} | pool {len(pairs):6d} "
                  f"| train {nt:5d} | eval {ne:4d}")


if __name__ == "__main__":
    main()
