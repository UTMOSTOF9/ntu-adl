# ADL homework2

## Steps for the submission file

### Download pretrained model

```bash
bash ./download.sh
```

### Run with pretrained model for submission file

```bash
bash ./run.sh /path/to/Taiwan-LLaMa-folder /path/to/adapter_checkpoint /path/to/input.json /path/to/output.json
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
gdown 1n0EY5c7nyzz_fhRdmAJJVojAM9GDUrZO
unzip hw3.zip
mv hw3/data .
rm -fr hw3 hw3.zip
```

### Run training script with docker

```bash
bash code/docker/run.sh code/scripts/train.sh

## inside docker

bash code/scripts/train_summarization.sh

```
