Ddavis=/gdata/lirui/dataset/DAVIS/data/
model_name='STM_SCL'
pre_model=/gdata/lirui/expdir/SCLVOS/group-stm-scl/log-sclstm-resnet50-iters200000-nce-t0.1-nce-k4096-expnamesclstm2021-09-15-05-11/checkpoints/davis_youtube_resnet50_99999.pth

outputdir1=/gdata/lirui/expdir/SCLVOS/group-stm-scl/log-sclstm-resnet50-iters200000-nce-t0.1-nce-k4096-expnamesclstm2021-09-15-05-11/visualize/vis
outputdir2=/gdata/lirui/expdir/SCLVOS/group-stm-scl/log-sclstm-resnet50-iters200000-nce-t0.1-nce-k4096-expnamesclstm2021-09-15-05-11/visualize/mask

python demo.py -g 0 -s val -y 17 -D $Ddavis -p $pre_model -model-name $model_name -output_viz_path $outputdir1 -output_mask_path $outputdir2
