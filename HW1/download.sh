mkdir -p data
gdown -O data/raw.tar.gz 19_c7uWXWFu7S2AtAOE--6Um9VLVxvNRd

tar xvzf data/raw.tar.gz -C data

bash scripts/_prepare_data.sh

echo "Downloaded and preprocessed data!"

gdown -O pretrained.tar.gz 1FXSNXm6vSzRhoBOX7NJuuTBcMINM-6yw

tar xvzf pretrained.tar.gz

echo "Downloaded pretrained models!"