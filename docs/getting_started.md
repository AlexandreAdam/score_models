# Getting started

### Installation

```bash
pip install score_models
```
## Neural network architectures
```python
from score_models import NCSNpp, MLP
```

### 1D
- Shape of the input is `[B, C]`, where `B` is an unspecified batch size.
- `C` is the number of channels of the input vector.
- `units` is the width of the hidden layers.
- `layers` is the number of hidden layers, not counting the attention bottleneck.
- `attention`: if True, use the attention mechanism in the bottleneck of the MLP
```python
net = MLP(C, units=100, layers=2, activation='silu', attention=True) # Example MLP
```


### Time-series, long sequences, etc.

- Shape of the input is `[B, C, L]`, where `L` is an unspecified sequence length.
- `dimensions=1` is used to specify 1D CNN layers in the architecture.
- `nf` is the base number of filters for the CNN layers.
- `ch_mult` is used to specify the number of levels in the U-net (`len(ch_mult)`. 
The entries serve as multiplicative factor for the number of filters at each level of the U-net (`ch_mult[i] * nf`). 
- `L` must be divisible by `2^len(ch_mult)`.
```python
net = NCSNpp(C, dimensions=1, nf=64, ch_mult=(1, 2, 4), attention=True) # Example NCSN++
```

### Images
- Shape of the input is `[B, C, H, W]`
- Both `H` and `W` must be divisible by `2^len(ch_mult)`
```python
net = NCSNpp(C, nf=128, ch_mult=(2, 2, 2, 2), attention=True) # Example NCSN++
```

### Videos/Voxels
- Shape of the input is `[B, C, H, W, D]`
- `H`, `W` and `D` must be divisible by `2^len(ch_mult)`
- `dimensions=3` is used to specify3D CNN layers in the architecture.
```python
net = NCSNpp(C, dimensions=3, nf=8, ch_mult=(1, 1, 2, 2), attention=True) # Example NCSN++
```

## Score-Based Model (SBM)
```python
from score_models import ScoreModel, VPSDE

sde = VPSDE()
sbm = ScoreModel(net, sde)
```

### Training
- `dataset` must be an instance of PyTorch `Dataset` or `DataLoader`. See e.g. [this tutorial](https
://pytorch.org/tutorials/beginner/basics/data_tutorial.html) to get started.
- If a `Dataset` instance is provided, we wrap it automatically with a `DataLoader`. Ideally, provide `batch_size` to the `fit` method in that case.
```python
sbm.fit(
    dataset,
    epochs=100,
    learning_rate=1e-3,
    batch_size=16,          # Only if dataset is not a torch DataLoader
    ema_decay=0.999,
    checkpoints_every=10,   # Save model every 10 epochs
    models_to_keep=1,       # Keep only the last model
    path='/path/to/checkpoints_directory',
    )
```

### Sampling
- `shape` of the input must be provided. `B` samples will be produced in parallel.
- `steps` specifies the discretization of the SDE. It can be increased to improve the sample quality.
```python
sbm.sample(shape=(B, C, ...), steps=1000)
```


### Saving and loading
```python
sbm.save('/path/to/checkpoints_directory')
sbm = ScoreModel(path='/path/to/checkpoints_directory')
```
