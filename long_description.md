# Score Models for Pytorch

A storage for score-based models. The `ScoreModel` interface gives access to the following utilities
- Simple initialisation of MLP, NCSN++ and DDPM neural network architectures
- A fit method to train the score model on a dataset using Denoising Score Matching (DSM).
- A sampling method based on an Euler-Maruyama discretisation of an SDE. 
- A simple interface to train an energy model using DSM

This repository is mainly intended for personal use. 
You might also want to refer to the original implementation at [https://github.com/yang-song/score_sde](https://github.com/yang-song/score_sde).


## Installation

To install the package, you can use pip:

```bash
pip install torch-score-models
```

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
