cd /gdata/lirui/project/vcl
python -m torch.distributed.launch --nproc_per_node=4 tools/data/prepare_youtube_flow.py --num-gpu 4