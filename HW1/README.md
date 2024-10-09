# ADL-HW1

## Environment Setup

- OS: Ubuntu 22.04

- Python 3.10

- Pytorch 2.1.0 (CUDA 12.1)

- Other requirements: see [requirements.txt](docker/requirements.txt)

Also, it's recommended to use docker for consistant enviroment to run the code without enviroment issues.

How to use please see [here](#docker)

---

## Reproduce the result (For TAs)

### Prepare data and pretrained

```bash
bash download.sh
```

### Run to get test result

```bash
bash run.sh <context.json> <test.json> <prediction.csv>
```

---

## Train by scripts (Optional)

See scripts to reproduce the pretrained.

1. Train bert-base-chinese (baseline)

```bash
bash scripts/train_bert-base-chinese.sh
```

2. Train hfl/chinese-macbert-base (variant bert)

```bash
bash scripts/train_hfl-chinese-macbert-base.sh
```

3. Train hfl/chinese-roberta-wwm-ext (variant bert)

```bash
bash scripts/train_hfl-chinese-roberta-wwm-ext.sh
```

4. Train bert-base-chinese from scratch (only qa task from scratch)

```bash
bash scripts/train_bert-base-chinese_qa_from-scratch.sh
```

5. Train end to end QA model with bert-base-chinese pretrained (end to end QA)

```bash
bash scripts/train_bert-base-chinese_end_to_end.sh
```

## Docker

You need install docker and nvidia-docker at first.

### Build docker image

```bash
cd docker
bash build.sh
cd -
```

### Run docker image

```bash
bash docker/run_docker.sh
# then you will enter the docker container and source code will mount to /code you can run scripts directly.
# like this:
bash scripts/train_bert-base-chinese.sh
```
