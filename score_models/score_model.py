from .base import ScoreModelBase


class ScoreModel(ScoreModelBase):
    def __init__(self, model: Union[str, Module] = None, checkpoint_directory=None, **hyperparameters):
        super().__init__(model, checkpoint_directory, **hyperparameters)

    def score(self, t, x):
        _, *D = x.shape
        return self.model(t=t, x=x) / sde.sigma(t).view(-1, *[1]*len(D))


class EnergyModel(ScoreModelBase):
    def __init__(self, model: Union[str, Module] = None, checkpoint_directory=None, **hyperparameters):
        super().__init__(model, checkpoint_directory, **hyperparameters)
    
    def energy(self, t, x):
        _, *D = x.shape
        return 0.5 * torch.sum((x - self.modelt(t=t, x=x))**2, dim=list(range(1, 1+len(D))))
    
    def score(self, t, x):
        return vmap(grad(self.energy, argnums=1))(t, x)
    
