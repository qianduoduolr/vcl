#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/test_prop.py --config $CONFIG --checkpoint $CHECKPOINT  --launcher pytorch ${@:4}

python $(dirname "$0")/test.py --config $CONFIG
