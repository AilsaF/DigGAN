# DigGAN

This repository contains the code of the following paper:

**DigGAN: Discriminator gradIent Gap Regularization for GAN Training with Limited Data** https://openreview.net/pdf?id=azBVn74t_2

**Abstract:** Generative adversarial nets (GANs) have been remarkably successful at learning to sample from distributions specified by a given dataset, particularly if the given dataset is reasonably large compared to its dimensionality. However, given limited data, classical GANs have struggled, and strategies like output-regularization, data-augmentation, use of pre-trained models and pruning have been shown to lead to improvements. Notably, the applicability of these strategies is 1) often constrained to particular settings, e.g., availability of a pretrained GAN; or 2) increases training time, e.g., when using pruning. In contrast, we propose a Discriminator gradIent Gap regularized GAN (DigGAN) formulation which can be added to any existing GAN. DigGAN augments existing GANs by encouraging to narrow the gap between the norm of the gradient of a discriminator’s prediction w.r.t. real images and w.r.t. the generated samples. We observe this formulation to avoid bad attractors within the GAN loss landscape, and we find DigGAN to significantly improve the results of GAN training when limited data is available


## Training

For DiffAugment-biggan-cifar (used to generate CIFAR10 and CIFAR100 in the paper):

Scrips are in DiffAugment-biggan-cifar/scripts. For instance, to run DigGAN on 100% CIFAR-10, one can use
```
CUDA_VISIBLE_DEVICES=0 python train.py \
--mirror_augment --use_multiepoch_sampler \
--which_best FID --num_inception_images 10000 \
--shuffle --batch_size 50 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 1000  \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 4000 --save_every 4000 --seed 0 \
--use_DigGAN --DigGAN_penalty_weight 1000
```

For PyTorch-StudioGAN-master (used to generate CUB200 and TINY-ImageNet in the paper):

Confgs are in PyTorch-StudioGAN-master/src/configs. To run configs, one can use command like
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/main.py -t -e -mpc -l -stat_otf -c src/configs/CUB200/BigGAN-DigGAN.json --eval_type "valid"
```


## Bibtex

Please cite our work if you find it useful for your research and work:

```
@ARTICLE{fang2022diggan,
         author = {Tiantian Fang and Ruoyu Sun and Alex Schwing},
         title = "{DigGAN: Discriminator gradIent Gap Regularization for GAN Training with Limited Data}",
         booktitle = {Conference on Neural Information Processing Systems},
         year = 2022,
}
```
