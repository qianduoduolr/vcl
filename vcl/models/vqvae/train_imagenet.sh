#!/usr/bin/env bash

python /gdata/lirui/project/vcl/vqvae/train_vqvae.py \
--downsample 2 --n_embed 4096 --train_epoch 3 --dataset imagenet --data_path /gdata2/pengjl/imagenet-lmdb/train \
 --in_c 256 --res_c 128 --emb_c 128 --save_path /gdata/lirui/models/vqvae

python /gdata/lirui/project/vcl/vqvae/train_vqvae.py \
--downsample 4 --n_embed 2048 --train_epoch 3 --dataset imagenet --data_path /gdata2/pengjl/imagenet-lmdb/train \
--in_c 256 --res_c 128 --emb_c 128 --save_path /gdata/lirui/models/vqvae
