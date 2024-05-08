import torch
import torch.nn as nn
import numpy as np


class JointScoreModel(nn.Module):

    def __init__(self, sde, models, x_shapes, model_uses, score_scale=1, **kwargs):
        super().__init__()
        self.sde = sde
        self.models = models
        self.x_shapes = x_shapes
        self.model_uses = model_uses
        self.score_scale = score_scale

    def split_x(self, x):
        B, D = x.shape

        # Split x into segments
        sub_x = []
        place = 0
        for shapex in self.x_shapes:
            sub_x.append(x[..., place : place + np.prod(shapex)].reshape(B, *shapex))
            place += np.prod(shapex)
        assert place == D
        return sub_x

    def join_x(self, sub_x):
        B = sub_x[0].shape[0]
        return torch.cat(tuple(S.reshape(B, -1) for S in sub_x), dim=-1)

    def forward(self, t, x, **kwargs):
        # Split x into segments
        sub_x = self.split_x(x)

        # Compute / store the score for each model
        scores = list(torch.zeros_like(sx) for sx in sub_x)
        for i, model in enumerate(self.models):
            # Select the segments from x that this model uses
            if self.model_uses[i] is None:
                model_x = x
                model_score = model(t, model_x, **kwargs)
            else:
                model_x = tuple(sub_x[j] for j in self.model_uses[i])
                # Compute the score for this model
                model_score = model(t, *model_x, **kwargs)

            # Ensure the score is a tuple
            if not isinstance(model_score, tuple):
                model_score = (model_score,)
            # Add the score to the appropriate segments of x (now stored in scores)
            if self.model_uses[i] is None:
                for j, score in enumerate(self.split_x(model_score[0])):
                    scores[j] += score
            else:
                for j, score in zip(self.model_uses[i], model_score):
                    scores[j] += score

        js = self.join_x(scores)
        return js * self.sde.sigma(t).view(-1, *[1] * len(js.shape[1:])) * self.score_scale
