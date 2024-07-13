# Sore-Models

This repository is a storage for score-based models. The `ScoreModel` interface
gives access to the following utilities

- Simple initialisation of MLP, NCSN++ and DDPM neural network architectures
- A fit method to train the score model on a dataset using Denoising Score
  Matching (DSM: see
  [Vincent 2011](https://www.iro.umontreal.ca/~vincentp/Publications/DenoisingScoreMatching_NeuralComp2011.pdf)).
- A sampling method based on an Euler-Maruyama discretisation of an SDE.

```{tableofcontents}

```
