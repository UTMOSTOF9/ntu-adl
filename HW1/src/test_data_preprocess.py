import argparse
import json
from pathlib import Path

SPLITS = ["train", "valid"]


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing the test data.")
    parser.add_argument(
        "--test_file", type=Path, default=None, help="Path to the test file."
    )
    parser.add_argument(
        "--context_file", type=Path, default=None, help="Path to the conext file."
    )
    parser.add_argument(
        "--output_folder", type=Path, default=None, help="Path to the output file."
    )
    args = parser.parse_args()
    return args


def preprocess_swag(data, context):
    swag = []

    for i in range(len(data)):
        # SWAG dataset format
        d = {}

        d['video-id'] = data[i]['id']
        d['fold-ind'] = str(0)
        d['startphrase'] = data[i]['question']

        d['sent1'] = data[i]['question']
        d['sent2'] = ''

        d['gold-source'] = 'gold'

        d['ending0'] = context[data[i]['paragraphs'][0]]
        d['ending1'] = context[data[i]['paragraphs'][1]]
        d['ending2'] = context[data[i]['paragraphs'][2]]
        d['ending3'] = context[data[i]['paragraphs'][3]]

        swag.append(d)
    return swag


def preprocess_squad_end2end(data, context):
    squad = []

    for i in range(len(data)):
        d = {}

        d['context'] = ''
        for paragraph_id in data[i]['paragraphs']:
            tmp = context[paragraph_id]
            if tmp[-1] != "。":
                tmp += "。"
            d['context'] += tmp

        d['id'] = data[i]['id']
        d['question'] = data[i]['question']
        d['title'] = 'test'

        squad.append(d)

    return squad


def main():
    data = json.loads(args.test_file.read_text())
    context = json.loads(args.context_file.read_text())

    swag = preprocess_swag(data, context)
    squad_end2end = preprocess_squad_end2end(data, context)

    swag_path = args.output_folder / "test_swag.json"
    swag_path.write_text(json.dumps(swag, indent=2, ensure_ascii=False, allow_nan=False), encoding='UTF-8')

    squad_end2end_path = args.output_folder / "test_squad_end2end.json"
    squad_end2end_path.write_text(
        json.dumps(squad_end2end, indent=2, ensure_ascii=False, allow_nan=False),
        encoding='UTF-8',
    )


if __name__ == "__main__":
    args = parse_args()
    main()
