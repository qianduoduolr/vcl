cd /ghome/lirui/project/sclvos/sclstm
echo 'run bash'

Ddavis=/gdata/lirui/dataset/DAVIS/data/
Dyoutube=/gdata/lirui/dataset/YouTube-VOS/
list_path=/gdata/lirui/dataset/YouTube-VOS/train/generated_frame_wise_meta.json
batch_size=1
num_workers=4
print_freq=4000

lr=0.01

nce_t=${1}
nce_k=${2}
total_iter=${3}
gpu=${4}

pre_model=/gdata/lirui/models/coco_pretrained_resnet50_679999_169.pth
expname=sclstm
output_dir=/gdata/lirui/expdir/SCLVOS/group-stm-scl
backbone=resnet50

python -m torch.distributed.launch --nproc_per_node $gpu train_davis_ssl.py  --batch-size $batch_size --print-freq $print_freq \
--num-workers $num_workers --total-iter $total_iter  --expname $expname --backbone $backbone --nce-t $nce_t --nce-k $nce_k \
--base-learning-rate $lr --multi true \
--pretrained-model $pre_model --Ddavis $Ddavis --Dyoutube $Dyoutube --output-dir $output_dir --list-path $list_path