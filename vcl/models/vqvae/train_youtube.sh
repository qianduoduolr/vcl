#!/usr/bin/env bash

python /gdata/lirui/project/vcl/vqvae/train_vqvae.py \
--downsample 4 --n_embed 8 --train_epoch 400 --dataset youtube --data_path /gdata/lirui/dataset/YouTube-VOS \
--list_path /gdata/lirui/dataset/YouTube-VOS/2018/train --in_c 256 --res_c 128 --emb_c 128 --save_path /gdata/lirui/models/vqvae

python /gdata/lirui/project/vcl/vqvae/train_vqvae.py \
--downsample 4 --n_embed 16 --train_epoch 400 --dataset youtube --data_path /gdata/lirui/dataset/YouTube-VOS \
--list_path /gdata/lirui/dataset/YouTube-VOS/2018/train --in_c 256 --res_c 128 --emb_c 128 --save_path /gdata/lirui/models/vqvae

python /gdata/lirui/project/vcl/vqvae/train_vqvae.py \
--downsample 4 --n_embed 32 --train_epoch 400 --dataset youtube --data_path /gdata/lirui/dataset/YouTube-VOS \
--list_path /gdata/lirui/dataset/YouTube-VOS/2018/train --in_c 256 --res_c 128 --emb_c 128 --save_path /gdata/lirui/models/vqvae


python /gdata/lirui/project/vcl/vqvae/train_vqvae.py \
--downsample 4 --n_embed 64 --train_epoch 400 --dataset youtube --data_path /gdata/lirui/dataset/YouTube-VOS \
--list_path /gdata/lirui/dataset/YouTube-VOS/2018/train --in_c 256 --res_c 128 --emb_c 128 --save_path /gdata/lirui/models/vqvae