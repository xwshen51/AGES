# Bidirectional Generative Modeling Using Adversarial Gradient Estimation

This repository contains the code for the paper *Bidirectional Generative Modeling Using Adversarial Gradient Estimation*.

## Install prerequisites
```
pip install -r requirements.txt
```

## Datasets
- Synthesized Mixture of Gaussians
- Stacked MNIST (same as that used in [Pac-GAN](https://arxiv.org/abs/1712.04086))
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [ImageNet](http://www.image-net.org)

## Run

- Train a bidirectional generative model (BGM) using *AGE-KL* on MoG:
```
sh run_mog_age_kl.sh
```
- Train a BGM using *AGE-ALL* on Stacked MNIST:
```
sh run_stack_mnist_age_all.sh
```
- Train a BGM using *AGE-ALL* on CelebA:
```
sh run_celeba_age_all.sh
```
- Train a BGM using *AGE-ALL* on ImageNet:
```
sh run_imagenet_age_all.sh
```
- Train a BGM using *AGE-KL* with *scaling clipping* on CelebA:
```
sh run_celeba_age_kl_sc.sh
```
- Train a unidirectional model using *AGE-ALL* on CelebA:
```
sh run_celeba_uni_age_all.sh
```

### Output
This will create a directory `./results/<dataset>/<save_name>` which will contain:

- **model.sav**: a Python distionary containing the generator, encoder, and discriminator.
- **gen.png**: generated images.
- **recon.png**: real images (odd columns) along with the reconstructions (even columns).
- **log.txt**: All losses computed during training.
- **config.txt**: training configurations.

### Help
Important arguments:

```
Model elements:
  --latent_dim          	dimension of the latent variable
  --prior               	prior distribution p_z of the latent variable
  --enc_dist {deterministic, gaussian, implicit}
                        	distribution of the encoder p_e(z|x) (default: gaussian)
  --dec_dist {deterministic, gaussian, implicit}
                        	distribution of the generator p_g(x|z) (default: deterministic)
                     
Objective:
  --div {all, kl, js, hellinger, revkl}
                        	use which divergence as the objective of generative modeling
  --unigen               	whether to train a unidirectional generative model (defalt: False)
  --clip               		whether to use the scaling clipping technique (defalt: False)
  --scale_lower          	lower bound of the scaling factor (default: 0.5)         
  --scale_upper          	upper bound of the scaling factor (default: None; use 1/scale_lower as the upper bound) 	

Datasets:
  --dataset {celeba, cifar, imagenet, mnist, mnist_stack, mog}
                        	name of the dataset (default: celeba)
  --data_dir          		directory of the dataset                       
  --image_size          	resolution of the image (default: 64)
```

## Acknowledgments
The code for SAGAN architectures is based on the PyTorch implementation of SAGAN from [this repository](https://github.com/voletiv/self-attention-GAN-pytorch).
