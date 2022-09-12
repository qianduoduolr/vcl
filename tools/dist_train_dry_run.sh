#!/usr/bin/env bash
key="exp_name"

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
CONFIGS=("none")
COUNT=0
DRY_RUN_FREQ=3
TASK_NUM=$3
TASK_FILE="/gdata/lirui/project/vcl/configs/train/ypb/task${TASK_NUM}.txt"
""
# export WANDB_API_KEY='ffddb91f64606cb17216362faa7bc29540061a69'
# export WANDB_CONFIG_DIR='/gdata/lirui/'
# export WANDB_CACHE_DIR='/gdata/lirui/'

for variable in {1..1000000}
do   
    echo 'start cycle'

    # check exp name
    while read line 
    do   
        TASK="/gdata/lirui/project/vcl/configs/train/ypb/$line.py"
        if [[ ${CONFIGS[@]/${TASK}/} != ${CONFIGS[@]} ]]
        then
            echo " "$TASK"  has run before ! "
            r=`expr $COUNT % $DRY_RUN_FREQ`
            if [ "$r" = "0" ]
            then 
                echo "run "$CONFIG""
                # main training (dry run)
                PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
                python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
                $(dirname "$0")/train_dry_run.py --config $CONFIG --launcher pytorch ${@:4}
            fi

        else
            echo "run "$TASK" "

            # main training
            PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
            python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
            $(dirname "$0")/train_dry_run.py --config $TASK --launcher pytorch ${@:4}

            # main test
            PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
            python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
            $(dirname "$0")/test_prop_dry_run.py --config $TASK --out-indices 2 --launcher pytorch ${@:4}

            PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
            python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
            $(dirname "$0")/test_prop_dry_run.py --config $TASK --out-indices 3 --launcher pytorch ${@:4}

            if [ $TASK = $CONFIG ]
            then
                echo "none"
            else
                CONFIGS+=($TASK)
            fi
        fi
        COUNT=`expr $COUNT + 1`
    done < $TASK_FILE

    echo "end cycle"
done