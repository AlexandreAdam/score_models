import torch
from torch.utils.data import TensorDataset
from score_models import ScoreModel, MLP

def test_training():
    # Create a dummy dataset
    # Assuming you have input features 'X' and target values 'y'
    X = torch.randn(10, 10)  # Replace with your input features

    # Convert the data into a TensorDataset
    dataset = TensorDataset(X)

    hyperparameters = {
        "input_dimensions": 10,
        "units": 10,
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