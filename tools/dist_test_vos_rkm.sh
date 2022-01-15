#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

for r in {8..30}
do   
    echo "run "$r" "
    R=`expr $r \* 2`
    # main test
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_prop.py --config $CONFIG --radius $R --out-indices 2 --launcher pytorch ${@:3}

    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test_prop.py --config $CONFIG --radius $R --out-indices 3 --launcher pytorch ${@:3}

    echo "end $R"
done