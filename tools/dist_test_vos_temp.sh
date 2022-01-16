#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
temps=(1 0.5 0.25 0.1 0.05)

for t in ${temps[@]}
do   
    echo "run "$t" "
    # main test
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_prop.py --config $CONFIG --temperature $t --out-indices 2 --launcher pytorch ${@:3}

    echo "end $t"
done