#!/usr/bin/env bash
# data processing
# cd /dev/shm
mkdir -p /dev/shm/2018/train
# cd 2018/train
cp -r /gdata/lirui/dataset/YouTube-VOS/2018/train/JPEGImages_s256.zip /dev/shm/2018/train/
unzip -d /dev/shm/2018/train/ /dev/shm/2018/train/JPEGImages_s256.zip
cp -r /gdata/lirui/dataset/YouTube-VOS/2018/train/Flows.zip /dev/shm/2018/train/
unzip -d /dev/shm/2018/train/ /dev/shm/2018/train/Flows.zip
cp -r /gdata/lirui/dataset/YouTube-VOS/2018/train/*.txt /dev/shm/2018/train/
echo "finish cp data"

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py --config $CONFIG --launcher pytorch ${@:3}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_prop.py --config $CONFIG --out-indices 2 --launcher pytorch ${@:3}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_prop.py --config $CONFIG --out-indices 3 --launcher pytorch ${@:3}