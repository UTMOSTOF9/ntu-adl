from pathlib import Path
import argparse
import json

SPLITS = ["train", "valid"]


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing the train / validation / test data.")
    parser.add_argument(
        "--data_dir", type=Path, default=None, help="A directory containing the training data."
    )
    parser.add_argument(
        "--context_file", type=Path, default=None, help="Path to the conext file."
    )
    parser.add_argument(
        "--output_folder", type=Path, default=None, help="A directory where output files are stored."
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

        for j in range(4):
            if (data[i]['relevant'] == data[i]['paragraphs'][j]):
                d['label'] = j

        swag.append(d)
    return swag


def preprocess_squad(data, context):
    squad = []

    for i in range(len(data)):
        # SQuAD dataset format
        d = {}

        d['answers'] = {}
        d['answers']['answer_start'] = [data[i]['answer']['start']]
        d['answers']['text'] = [data[i]['answer']['text']]

        d['context'] = context[data[i]['relevant']]
        d['id'] = data[i]['id']

        d['question'] = data[i]['question']
        d['title'] = 'train'

        squad.append(d)
    return squad

def preprocess_squad(data, context):
    squad = []

    for i in range(len(data)):
        # SQuAD dataset format
        d = {}

        d['answers'] = {}
        d['answers']['answer_start'] = [data[i]['answer']['start']]
        d['answers']['text'] = [data[i]['answer']['text']]

        d['context'] = context[data[i]['relevant']]
        d['id'] = data[i]['id']

        d['question'] = data[i]['question']
        d['title'] = 'train'

        squad.append(d)
    return squad

def preprocess_squad_end2end(data, context):
    squad = []

    for i in range(len(data)):
        # SQuAD dataset format
        d = {}

        d['answers'] = {}
        d['context'] = ''
        d['answers']['text'] = [data[i]['answer']['text']]

        d['answers']['answer_start'] = 0
        for paragraph_id in data[i]['paragraphs']:
            tmp = context[paragraph_id]
            if tmp[-1] != "。":
                tmp += "。"
            d['context'] += tmp
            if paragraph_id != data[i]['relevant']:
                d['answers']['answer_start'] += len(tmp)
            else:
                d['answers']['answer_start'] += data[i]['answer']['start']
        d['answers']['answer_start'] = [d['answers']['answer_start']]
        d['id'] = data[i]['id']

        d['question'] = data[i]['question']
        d['title'] = 'train'

        squad.append(d)
    return squad

def main():
    args.output_folder.mkdir(parents=True, exist_ok=True)

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}

    context_path = args.data_dir / f"context.json"
    context = json.loads(context_path.read_text())

    for split in SPLITS:
        swag = preprocess_swag(data[split], context)
        squad = preprocess_squad(data[split], context)
        squad_end2end = preprocess_squad_end2end(data[split], context)

        swag_path = args.output_folder / f"{split}_swag.json"
        swag_path.write_text(json.dumps(swag, indent=2, ensure_ascii=False, allow_nan=False), encoding='UTF-8')

        squad_path = args.output_folder / f"{split}_squad.json"
        squad_path.write_text(json.dumps(squad, indent=2, ensure_ascii=False, allow_nan=False), encoding='UTF-8')

        squad_path = args.output_folder / f"{split}_squad_end2end.json"
        squad_path.write_text(json.dumps(squad_end2end, indent=2, ensure_ascii=False, allow_nan=False), encoding='UTF-8')


if __name__ == "__main__":
    args = parse_args()
    main()
