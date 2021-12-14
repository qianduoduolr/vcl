#!/usr/bin/env bash

mkdir -p /dev/shm/2018/train
mkdir -p  /dev/shm/2018/train_all_frames
# cd 2018/train
cp -r /gdata/lirui/dataset/YouTube-VOS/2018/train/zip_data/JPEGImages_s256.zip /dev/shm/2018/train/
unzip -d /dev/shm/2018/train/ /dev/shm/2018/train/JPEGImages_s256.zip
cp -r /gdata/lirui/dataset/YouTube-VOS/2018/train_all_frames/zip_data/Flows_s256.zip /dev/shm/2018/train_all_frames/
unzip -d /dev/shm/2018/train_all_frames/ /dev/shm/2018/train_all_frames/Flows_s256.zip
cp -r /gdata/lirui/dataset/YouTube-VOS/2018/train/zip_data/Annotations_s256.zip /dev/shm/2018/train/
unzip -d /dev/shm/2018/train/ /dev/shm/2018/train/Annotations_s256.zip
cp -r /gdata/lirui/dataset/YouTube-VOS/2018/train/*.txt /dev/shm/2018/train/
cp -r /gdata/lirui/dataset/YouTube-VOS/2018/train/*.json /dev/shm/2018/train/
echo "finish cp data"

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

for variable in {1..100000}
do   
    echo 'start training'
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/train_dry_run.py --config $CONFIG --launcher pytorch ${@:3}

    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/test_prop_dry_run.py --config $CONFIG --out-indices 2 --launcher pytorch ${@:3}

    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/test_prop_dry_run.py --config $CONFIG --out-indices 3 --launcher pytorch ${@:3}
        
    echo "$variable finished !"
done  
