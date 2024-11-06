docker run --gpus all \
    --shm-size=32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ipc=host \
    --net=host \
    -v $PWD:/code \
    -v ~/.cache:/root/.cache \
    -v /home/jack/.netrc:/root/.netrc \
    -it --rm adl:HW3 $@