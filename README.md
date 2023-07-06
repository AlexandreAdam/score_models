# Score Models for Pytorch

[![PyPI version](https://badge.fury.io/py/score_models.svg)](https://badge.fury.io/py/score_models)
[![codecov](https://codecov.io/gh/AlexandreAdam/torch_score_models/branch/dev/graph/badge.svg)](https://codecov.io/gh/AlexandreAdam/torch_score_models)

A storage for score-based models. The `ScoreModel` interface gives access to the following utilities
- Simple initialisation of MLP, NCSN++ and DDPM neural network architectures
- A fit method to train the score model on a dataset using Denoising Score Matching (DSM: see [Vincent 2011](https://www.iro.umontreal.ca/~vincentp/Publications/DenoisingScoreMatching_NeuralComp2011.pdf)).
- A sampling method based on an Euler-Maruyama discretisation of an SDE. 
- A simple interface to train an energy model using DSM

This repository is mainly intended for personal use. 
You might also want to refer to the original implementation at [https://github.com/yang-song/score_sde](https://github.com/yang-song/score_sde).

## Installation

To install the package, you can use pip:

```bash
pip install score_models
```

## Usage


### ScoreModel

The `ScoreModel` class is the main interface for training and using score models, defined as

```math
\mathbf{s}_\theta(t, \mathbf{x}) \equiv \nabla_\mathbf{x} \log p_t(\mathbf{x}) = \frac{1}{\sigma(t)} f_\theta (t, \mathbf{x})$$
```

where $\sigma(t)$ is the standard deviation of the perturbation kernel $`p_t(\mathbf{x} \mid \mathbf{x}_0)`$
of an SDE and $f_\theta : [0, 1] \times\mathbb{R}^d \to \mathbb{R}^d$ is a neural network for $\mathbf{x} \in \mathbb{R}^d$. 

The `ScoreModel` class extends the `torch.nn.Module` class. Example usage:

```python
from score_models import ScoreModel, EnergyModel, NCSNpp, MLP, DDPM

# Create a ScoreModelBase instance with Yang Song's NCSN++ architecture and the VESDE
net = NCSNpp(channels=1, nf=128, ch_mult=[2, 2, 2, 2])
model = ScoreModelBase(model=net, sigma_min=1e-2, sigma_max=50, device="cuda")
# ... or the VPSDE
model = ScoreModelBase(model=net, beta_min=1e-2, beta_max=20, device="cuda")

# NN Architectures support a Unet with 1D convolutions for time series input data
net = NCSNpp(channels=1, nf=128, ch_mult=[2, 2, 2, 2], dimensions=1)
# ... or 3D convolutions for videos/voxels
net = NCSNpp(channels=1, nf=128, ch_mult=[2, 2, 2, 2], attention=False, dimensions=3)
# You can also use a simpler MLP architecture 
net = MLP(dimensions=dim, layers=4, units=100)
# ... or Jonathan Ho's DDPM architecture
net = DDPM(channels=1, dimensions=2, nf=128, ch_mult=[2, 2, 2, 2])

# Train the model
model.fit(dataset=your_dataset, epochs=100, learning_rate=1e-4)

# Generate samples from the trained model
samples = model.sample(size=[batch_size, *dimensions], N=1000)

# Compute the score for a given input
score = model.score(t, x)

# Initialise the score model and its neural network from a path to a checkpoint directory 
score = ScoreModel(checkpoint_directory)
```

### EnergyModel

The `EnergyModel` class works in pretty much the same way as `ScoreModel`, but implements the score via the 
automatic differentation of an energy model
```math
E_\theta(t, \mathbf{x}) = \frac{1}{2 \sigma(t)} \lVert \mathbf{x} - f_\theta (t, \mathbf{x}) \rVert_2^ 2
```

This is to say that the score is defined as
```math
\mathbf{s}_\theta (t, \mathbf{x}) = - \nabla_\mathbf{x} E_\theta(t, \mathbf{x}) 
```


**Note**: When using the MLP architecture, the energy model can be constructed more efficiently as the output of the
neural network by specifying `nn_is_energy` in the hyperparameters of the MLP, which will modify the neural network 
architecture to be a function $f_\theta: [0, 1] \times\mathbb{R}^d \to \mathbb{R}$ instead of $f_\theta: [0, 1] \times\mathbb{R}^d \to \mathbb{R}^d$. An `output_activation` like `relu` 
can also be specified to make the energy positive or bounded from below.


### Training Parameters

When training the model using the `fit` method, you can provide various parameters to customize the training process. Some important parameters include:

- `dataset`: The training dataset (`torch.utils.data.Dataset`).
- `epochs`: The number of training epochs.
- `learning_rate`: The learning rate for the ADAM optimizer.
- `batch_size`: The batch size for training.
- `checkpoints_directory`: The directory to save model checkpoints (default: None).
- `seed`: The random seed for numpy and torch.

Refer to the method's docstring or the class definition for more details on available parameters.

## Citations

If you use this package in your research, please consider citing the following papers:

```bibtex
@inproceedings{NEURIPS2020_4c5bcfec,
    author      = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
    pages       = {6840--6851},
    publisher   = {Curran Associates, Inc.},
    title       = {Denoising Diffusion Probabilistic Models},
    url         = {https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca

4b-Paper.pdf},
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

---

This package is licensed under the MIT License.

