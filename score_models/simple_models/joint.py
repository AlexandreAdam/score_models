import torch
import torch.nn as nn
import numpy as np


class JointScoreModel(nn.Module):
    """
    A score model which combines the scores from multiple models.

    This score model class allows for multiple score models to combine their
    scores arbitrarily. They may share all, some, or none of the model space
    with the class handling the bookkeeping. The scores from each model (where
    they use the same model dimensions) are simply summed. The class may also
    handle multiple inputs, internally they are combined into a single massive
    concatenated ``x`` vector, when passed to the models the ``x`` vector is
    split into the appropriate segments ``x_0, x_1, ..., x_n`` and each one is
    converted into the expected shape (defined by the ``x_shapes`` argument).

    Args:
        sde: The SDE that the score model is associated with.
        models: A list of score models.
        x_shapes: A list of shapes for the x vectors that the models expect.
            These are the shapes that the flat-concatenated ``x`` vector will
            be split into.
        model_uses: A list of lists of integers, where each list is the indices
            of the x vectors corresponding to ``x_shapes`` that each model uses.
            If None, the model will be passed the full ``x`` vector.
    """

    def __init__(self, sde, models, x_shapes, model_uses, **kwargs):
        super().__init__()
        self.sde = sde
        self.models = models
        self.x_shapes = x_shapes
        self.model_uses = model_uses

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

    @property
    def xsize(self):
        return sum(np.prod(shapex) for shapex in self.x_shapes)

    def forward(self, t, x, **kwargs):
        print("in joint", x.shape, self.x_shapes)
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
            print("model_score", model_score)
            if self.model_uses[i] is None:
                for j, score in enumerate(self.split_x(model_score[0])):
                    scores[j] += score
            else:
                for j, score in zip(self.model_uses[i], model_score):
                    scores[j] += score

        js = self.join_x(scores)
        return js * self.sde.sigma(t).view(-1, *[1] * len(js.shape[1:]))
