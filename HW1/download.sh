mkdir -p data
gdown -O data/raw.tar.gz 1MNnsPFz79TC6zahCQ9_0FIgzNrTNQrEf

tar xvzf data/raw.tar.gz -C data

bash scripts/_prepare_train_dataset.sh

echo "Downloaded data and preprocessed training data!"

gdown -O pretrained.tar.gz 1-CQjidj44odX3ZpqOE7u7RRNk_SYmsi_

tar xvzf pretrained.tar.gz