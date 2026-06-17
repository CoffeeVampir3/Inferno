import csv
import os

DATA = "abliteration_data"
N_HARMLESS_EVAL = 256


def load(path, cap=None):
    out = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row:
                continue
            text = " ".join(row[0].split())
            if not text:
                continue
            out.append(text)
            if cap is not None and len(out) >= cap:
                break
    return out


def write(name, lines):
    with open(os.path.join(DATA, name), "w") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    harmful_train = load(f"{DATA}/harmful/data/train-00000-of-00001.csv")
    harmless_train = load(
        f"{DATA}/harmless/data/train-00000-of-00001.csv", cap=len(harmful_train)
    )
    harmful_eval = load(f"{DATA}/harmful/data/test-00000-of-00001.csv")
    harmless_eval = load(
        f"{DATA}/harmless/data/test-00000-of-00001.csv", cap=N_HARMLESS_EVAL
    )

    write("harmful_train.txt", harmful_train)
    write("harmless_train.txt", harmless_train)
    write("harmful_eval.txt", harmful_eval)
    write("harmless_eval.txt", harmless_eval)

    print(f"harmful_train  : {len(harmful_train)}")
    print(f"harmless_train : {len(harmless_train)}")
    print(f"harmful_eval   : {len(harmful_eval)}")
    print(f"harmless_eval  : {len(harmless_eval)}")


if __name__ == "__main__":
    main()
