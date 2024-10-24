# ADL homework2

## Steps for the submission file

### Download pretrained model

```bash
bash ./download.sh
```

### Run with pretrained model for submission file

```bash
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```

## Steps for reproducing the result with docker

### build docker first

```bash
cd code/docker
bash build.sh
cd -
```

### Put data into the correct path

Please put the data into the correct path as follows:

```bash
gdown 1YWrSX37OfQSDJgdaeyrrh6y9110I1iPi
unzip data_for_train.zip
```

### Run training script with docker

```bash
bash code/docker/run.sh

## inside docker

bash code/scripts/train_summarization.sh

```
