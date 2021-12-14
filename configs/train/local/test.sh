#!/bin/bash 
CONFIG="/gdata/lirui/project/vcl/configs/train/ypb/train_dry_run.py"
CONFIG_RUN="none"
CONFIGS=("none")
key="exp_name"
COUNT=0
DRY_RUN_FREQ=2

for variable in {1..3}
do   
     echo 'start training'

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
               fi
          else
               echo "run "$TASK" "
               if [ $TASK = $CONFIG ]
               then
                    echo "none"
               else
                    CONFIGS+=($TASK)
               fi
          fi
          COUNT=`expr $COUNT + 1`
     done < /home/lr/project/vcl/configs/train/local/task.txt

     echo "finished"

done  