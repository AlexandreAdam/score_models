import torch
from torch.utils.data import TensorDataset
from score_models import ScoreModel, EnergyModel, MLP, NCSNpp
import shutil, os

class Dataset(torch.utils.data.Dataset):
    def __init__(self, size, channels, dimensions:list, conditioning="None", test_input_list=False):
        self.size = size
        self.channels = channels
        self.dimensions = dimensions
        self.conditioning = conditioning
        self.test_input_list = test_input_list

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.test_input_list:
            return torch.randn(self.channels, *self.dimensions),
        if self.conditioning.lower() == "none":
            return torch.randn(self.channels, *self.dimensions)
        elif self.conditioning.lower() == "time":
            return torch.randn(self.channels, *self.dimensions), torch.randn(self.channels)
        elif self.conditioning.lower() == "input":
            return torch.randn(self.channels, *self.dimensions), torch.randn(self.channels, *self.dimensions)

# def test_training_conditioning_input_ncsnpp():
    # C = 1
    # D = 16
    # dim = 2
    # B = 5
    # size = 2*B
    # dataset = Dataset(size, C, [D]*dim, conditioning="input")
    # net = NCSNpp(

        

def test_training_score_mlp():
    C = 10
    B = 5
    size = 2*B
    dataset = Dataset(size, C, [])
    hyperparameters = {
        "dimensions": C,
        "units": 2*C,
        "layers": 2,
        "time_embedding_dimensions": 32,
        "embedding_scale": 32,
        "activation": "swish",
        "time_branch_layers": 1
    }
    net = MLP(**hyperparameters)
    # Create an instance of ScoreModel
    model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=10)

    # Define any preprocessing function if needed
    def preprocessing_fn(x):
        return x

    # Set the hyperparameters and other options for training
    learning_rate = 1e-3
    ema_decay = 0.9999
    batch_size = 1
    epochs = 10
    epsilon = 0 # avoid t=0 in sde sample (not needed for VESDE)
    warmup = 0 # learning rate warmup
    clip = 0. # gradient clipping
    checkpoints_directory = os.path.dirname(os.path.abspath(__file__)) + "/checkpoints"
    seed = 42

    # Fit the model to the dataset
    losses = model.fit(
        dataset, 
        preprocessing_fn=preprocessing_fn, 
        learning_rate=learning_rate, 
        ema_decay=ema_decay,
        batch_size=batch_size, 
        epochs=epochs, 
        epsilon=epsilon, 
        warmup=warmup, 
        clip=clip, 
        checkpoints_directory=checkpoints_directory, 
        seed=seed
        )
    print(losses)
    # leave the checpoints there until next test

def test_training_score_mlp_input_list():
    C = 10
    B = 5
    size = 2*B
    dataset = Dataset(size, C, [], test_input_list=True)
    checkpoints_directory = os.path.dirname(os.path.abspath(__file__)) + "/checkpoints"
    model = ScoreModel(checkpoints_directory=checkpoints_directory)
    losses = model.fit(
        dataset, 
        checkpoints_directory=checkpoints_directory, 
        epochs=10,
        batch_size=1
        )


def test_training_load_checkpoint():
    C = 10
    B = 5
    size = 2*B
    dataset = Dataset(size, C, [])
    checkpoints_directory = os.path.dirname(os.path.abspath(__file__)) + "/checkpoints"
    model = ScoreModel(checkpoints_directory=checkpoints_directory)
    losses = model.fit(
        dataset, 
        checkpoints_directory=checkpoints_directory, 
        epochs=10,
        batch_size=1
        )
    shutil.rmtree(checkpoints_directory)

    
def test_training_score_ncsnpp():
    C = 1
    D = 140
    B = 5
    size = 2*B
    dataset = Dataset(size, C, [D])
    hyperparameters = {
     'channels': C,
     'nf': 8,
     'activation_type': 'swish',
     'ch_mult': (2, 2),
     'num_res_blocks': 2,
     'resample_with_conv': True,
     'dropout': 0.0,
     'fir': True,
     'fir_kernel': (1, 3, 3, 1),
     'skip_rescale': True,
     'progressive': 'output_skip',
     'progressive_input': 'input_skip',
     'init_scale': 0.01,
     'fourier_scale': 16.0,
     'resblock_type': 'biggan',
     'combine_method': 'sum',
     'attention': True,
     'dimensions': 1,
     'sde': 'vesde',
     'sigma_min': 0.001,
     'sigma_max': 200,
     'epsilon': 0.0,
     'T': 1.0}
    net = NCSNpp(**hyperparameters)
    # Create an instance of ScoreModel
    model = ScoreModel(model=net, sigma_min=1e-2, sigma_max=10)

    # Define any preprocessing function if needed
    def preprocessing_fn(x):
        return x

    # Set the hyperparameters and other options for training
    learning_rate = 1e-3
    ema_decay = 0.9999
    batch_size = 1
    epochs = 2
    epsilon = 0 # avoid t=0 in sde sample (not needed for VESDE)
    warmup = 0 # learning rate warmup
    clip = 0. # gradient clipping
    checkpoints_directory = None
    seed = 42

    # Fit the model to the dataset
    losses = model.fit(
        dataset, 
        preprocessing_fn=preprocessing_fn, 
        learning_rate=learning_rate, 
        ema_decay=ema_decay,
        batch_size=batch_size, 
        epochs=epochs, 
        epsilon=epsilon, 
        warmup=warmup, 
        clip=clip, 
        checkpoints_directory=checkpoints_directory, 
        seed=seed
        )
    print(losses)

def test_training_energy():
    # Create a dummy dataset
    X = torch.randn(10, 10)

    # Convert the data into a TensorDataset
    dataset = TensorDataset(X)

    hyperparameters = {
        "dimensions": 10,
        "units": 10,
        "layers": 2,
        "time_embedding_dimensions": 32,
        "embedding_scale": 32,
        "activation": "swish",
        "time_branch_layers": 1,
        # "nn_is_energy": True
    }
    net = MLP(**hyperparameters)
    # Create an instance of ScoreModel
    model = EnergyModel(model=net, sigma_min=1e-2, sigma_max=10)

    # Define any preprocessing function if needed
    def preprocessing_fn(x):
        return x

    # Set the hyperparameters and other options for training
    learning_rate = 1e-3
    ema_decay = 0.9999
    batch_size = 1
    epochs = 10
    epsilon = 0 # avoid t=0 in sde sample (not needed for VESDE)
    warmup = 0 # learning rate warmup
    clip = 0. # gradient clipping
    checkpoints_directory = None
    seed = 42

    # Fit the model to the dataset
    losses = model.fit(
        dataset, 
        preprocessing_fn=preprocessing_fn, 
        learning_rate=learning_rate, 
        ema_decay=ema_decay,
        batch_size=batch_size, 
        epochs=epochs, 
        epsilon=epsilon, 
        warmup=warmup, 
        clip=clip, 
        checkpoints_directory=checkpoints_directory, 
        seed=seed
        )
    print(losses)


