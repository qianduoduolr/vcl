Ddavis=/braindat/lab/hubo/DATASET/DAVIS/
Dyoutube=/braindat/lab/hubo/DATASET/Youtube-VOS/
list_path=/braindat/lab/hubo/DATASET/Youtube-VOS//train/generated_frame_wise_meta.json
batch_size=1
num_workers=8
print_freq=4000

nce_t=${1}
nce_k=${2}

gpu=${3}

pre_model=//braindat/lab/hubo/CODE/sclstm/models/coco_pretrained_resnet50_679999_169.pth
expname=sclstm
output_dir=./output
backbone=resnet50

python -m torch.distributed.launch --nproc_per_node $gpu train_davis.py  --batch-size $batch_size --print-freq $print_freq \
--num-workers $num_workers  --expname $expname --backbone $backbone --nce-t $nce_t --nce-k $nce_k \
--multi true \
--pretrained-model $pre_model --Ddavis $Ddavis --Dyoutube $Dyoutube --output-dir $output_dir --list-path $list_path