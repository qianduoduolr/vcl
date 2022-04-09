#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}
CONFIGS=("none")
COUNT=0
DRY_RUN_FREQ=3

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
                $(dirname "$0")/train_dry_run.py --config $CONFIG --launcher pytorch ${@:3}
            fi

        else
            echo "run "$TASK" "

            # main training
            cd /gdata/lirui/project/moco
            python main_moco.py --config $TASK

            if [ $TASK = $CONFIG ]
            then
                echo "none"
            else
                CONFIGS+=($TASK)
            fi
        fi
        COUNT=`expr $COUNT + 1`
    done < /gdata/lirui/project/vcl/configs/train/ypb/task_moco.txt

    echo "end cycle"
done
