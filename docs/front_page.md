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

This package is intended to
1. Store neural network architectures useful for score-based models (SBM);
2. Provide methods to train SBM and sample from them;
3. Provide utilities for SBM.

This package is not intended to be a general-purpose machine learning library nor a general-purpose sampler package.
If you use it for your science, please cite the original authors of the methods you use.

## Table of Content

```{tableofcontents}
```
