# Score-Models
[![PyPI version](https://badge.fury.io/py/score_models.svg)](https://badge.fury.io/py/score_models)
[![codecov](https://codecov.io/gh/AlexandreAdam/torch_score_models/branch/master/graph/badge.svg)](https://codecov.io/gh/AlexandreAdam/score_models)


This package started a repository for personal use to avoid code duplication across different projects.
I used it to store neural network architectures and some useful methods 
bootlegged from the original implementation of the Score-Based Generative Modeling repository by Yang Song.
Over time, I added some stuff and cleaned up the codebase, with the goal to collect useful methods in a 
single package and to provide a simple interface to train and use score models.

If you want to contribute or would like to see some functionalities added,
please feel free to open an issue or a pull request. 
You can use the Github Button at the top of the page to access the repository or to open an issue.

## Scope of the package

The interface of this package is centered around providing basic utilities for score-based models (SBM)
```python
from score_models import NCSNpp, VPSDE, ScoreModel

C = 1 # number of channels
D = (64, 64) # spatial dimensions
sde = VPSDE()
net = NCSNpp(channels=C, dimensions=len(D), nf=128, ch_mult=[2, 2, 2, 2])
model = ScoreModel(net=net, sde=sde)
```
### It can be used to...
#### Train SBM
```python
model.fit(
    dataset, 
    epochs=100, 
    path="/path/to/checkpoint/directory",
    checkpoint_every=10
    )
```
#### Sample from SBM
```python
B = 10 # batch size
samples = model.sample(shape=(B, C, *D), steps=1000)
```

#### Save and Load SBM
```python
# Save the model
model.save("/path/to/checkpoint/directory")
# Reload the model
model = ScoreModel(path="/path/to/checkpoint/directory")
```

#### Evalute log-likelihood
```python
log_likelihood = model.log_likelihood(samples)
```

### And some more
See the [Getting Started](getting_started.md) page for more information.


### This package is not...
intended to be a general-purpose machine learning library nor a general-purpose sampler package.


## Table of Content

```{tableofcontents}
```
