import argparse
import json
from pathlib import Path

import jsonlines


def load_jsonl(fpath):
    with jsonlines.open(fpath, 'r') as f:
        to_json = [obj for obj in f]
    return to_json


def save_jsonl(object, out_fpath):
    with jsonlines.open(out_fpath, 'w') as f:
        f.write_all(object)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    data = load_jsonl(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


main()
