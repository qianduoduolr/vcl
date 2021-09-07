cd /ghome/lirui/project/sclvos/stm
echo 'run bash'

Ddavis=/gdata/lirui/dataset/DAVIS/data/
Dyoutube=/gdata/lirui/dataset/YouTube-VOS/
batch_size=1
num_workers=4

eval_freq=${1}
total_iter=${2}
gpu=${3}

pre_model=/gdata/lirui/models/coco_pretrained_resnet50_679999_169.pth
expname=reproduce_stm
output_dir=/gdata/lirui/expdir/SCLVOS/group-stm-scl
backbone=resnet50

python -m torch.distributed.launch --nproc_per_node $gpu train_davis.py  --batch-size $batch_size \
--num-workers $num_workers --total-iter $total_iter  --eval-freq $eval_freq  --expname $expname --backbone $backbone \
--multi true \
--pretrained-model $pre_model --Ddavis $Ddavis --Dyoutube $Dyoutube --output-dir $output_dir