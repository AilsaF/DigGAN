#!/bin/bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 src/main.py -t -e -mpc -l -stat_otf -c src/configs/CUB200/BigGAN-DigGAN.json --eval_type "valid" --exp_name "CUB200-gradclose1000"
# --reduce_train_dataset 0.1

