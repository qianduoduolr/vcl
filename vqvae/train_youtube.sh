#!/usr/bin/env bash

python /gdata/lirui/project/vcl/vqvae/train_vqvae.py \
--downsample 4 --n_embed 2048 --train_epoch 400 --dataset youtube --data_path /gdata/lirui/dataset/YouTube-VOS \
--list_path /gdata/lirui/dataset/YouTube-VOS/2018/train --in_c 512 --res_c 128 --emb_c 512 --save_path /gdata/lirui/models/vqvae

python /gdata/lirui/project/vcl/vqvae/train_vqvae.py \
--downsample 2 --n_embed 4096 --train_epoch 400 --dataset youtube --data_path /gdata/lirui/dataset/YouTube-VOS \
--list_path /gdata/lirui/dataset/YouTube-VOS/2018/train --in_c 512 --res_c 128 --emb_c 512 --save_path /gdata/lirui/models/vqvae
