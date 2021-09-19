Ddavis=/gdata/lirui/dataset/DAVIS/data/
model_name='STM_SCL'
pre_model=/gdata/lirui/expdir/SCLVOS/group-stm-scl/log-sclstm-resnet50-iters200000-nce-t0.1-nce-k4096-expnamesclstm2021-09-15-05-11/checkpoints/davis_youtube_resnet50_99999.pth

outputdir=/gdata/lirui/expdir/SCLVOS/group-stm-scl/log-sclstm-resnet50-iters200000-nce-t0.1-nce-k4096-expnamesclstm2021-09-15-05-11

python eval.py -g 0 -s val -y 17 -D $Ddavis -p $pre_model -backbone resnet50 -model-name $model_name -output-dir $outputdir