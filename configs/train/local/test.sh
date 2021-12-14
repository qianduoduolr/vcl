#!/bin/bash 
key="exp_name"
CONFIG="none"
for variable in {1..100000}
do
     while read line 
     do   
          n=${line: 0 : 8 }
          if [ "$n" = "$key" ];then
               CONFIG=$line
               break	
          fi
     done < /home/lr/project/vcl/configs/train/local/train_dry_run.py
     CONFIG="/gdata/lirui/project/vcl/configs/train/ypb/"${CONFIG:12:-1}".py"
     echo $CONFIG
done