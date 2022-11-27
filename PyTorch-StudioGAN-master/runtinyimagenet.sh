#!/bin/bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 src/main.py -t -e -mpc -l -stat_otf -c src/configs/TINY_ILSVRC2012/BigGAN-DigGAN-DiffAug.json --eval_type "valid"

