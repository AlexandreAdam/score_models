# Score Models for torch

A storage for score-based model. The trained models will be able to generate images solving a Variance Exploding SDE using an Euler-Maruyama solver. More importantly, they give us access to the score of the implicit distirbution defined by the dataset and the forward process of the VESDE.  

This repository is mainly intended for personal use. You might also want to refer to the original implementation [https://github.com/yang-song/score_sde](url). 

## Installation
To install, first clone this repository, then run the following commands
```
cd score_models
python setupy.py install
```

## Usage
### Training a score model

Default parameters for DDPM might be
```
{
  "channels": 1,
  "nf": 128,
  "image_size": 256,
  "activation_type": "relu",
  "ch_mult": [1, 1, 2, 2, 4, 4],
  "num_res_blocks": 2,
  "resample_with_conv": true,
  "dropout": 0,
  "sigma_min": 1e-2,
  "sigma_max": 50
}
```
For NCSNpp, they could be
```
{
  "channels": 1,
  "nf": 128,
  "image_size": 256,
  "ch_mult": [1, 1, 2, 2, 2, 2, 2],
  "num_res_blocks": 2,
  "activation_type": "swish",
  "dropout": 0.0,
  "resample_with_conv": true,
  "fir": true,
  "fir_kernel": [1, 3, 3, 1],
  "skip_rescale": true,
  "progressive": "output_skip",
  "progressive_input": "input_skip",
  "init_scale": 1e-2,
  "fourier_scale": 16.0,
  "resblock_type": "biggan",
  "combine_method": "sum",
  "sigma_min": 1e-2,
  "sigma_max": 50
}
```
Of course, the two most important parameters are ``sigma_min`` and ``sigma_max``, which will depend on the dataset being modeled. These 
hyperparameters should be saved on a ``.json`` file, then passed to the training script.

Usage of the training script might look like this
```
python train_score_model.py\
  --model_architecture=ddpm\
  --dataset_path=dataset.h5\
  --dataset_key=the_h5_key\
  --model_parameters=config.json\
  --epochs=10000\
  --learning_rate=2e-5\
  --max_time=70\
  --batch_size=16\
  --logdir=logs/\
  --logname_prefixe=ddpm\
  --model_dir=models/\
  --checkpoints=10\
  --seed=42
```


## Citations
```bibtex
@inproceedings{NEURIPS2020_4c5bcfec,
    author      = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
    pages       = {6840--6851},
    publisher   = {Curran Associates, Inc.},
    title       = {Denoising Diffusion Probabilistic Models},
    url         = {https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf},
    volume      = {33},
    year        = {2020}
}
```

```bibtex
@inproceedings{song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```
