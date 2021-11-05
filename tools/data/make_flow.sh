GPU=$1

cd /gdata/lirui/project/vcl
python -m torch.distributed.launch --nproc_per_node=$GPU tools/data/prepare_youtube_flow.py --num-gpu $GPU