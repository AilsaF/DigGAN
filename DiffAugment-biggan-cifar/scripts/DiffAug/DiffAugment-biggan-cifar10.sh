#!/bin/bash
CUDA_VISIBLE_DEVICES=4 python train.py --DiffAugment translation,cutout \
--mirror_augment --use_multiepoch_sampler \
--which_best FID --num_inception_images 50000 \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 1000 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 4000 --save_every 4000 --seed 0 \
--num_best_copies 5 --num_save_copies 2