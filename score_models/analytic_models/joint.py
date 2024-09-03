from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from ..sde import SDE
from ..sbm import ScoreModel


class JointScoreModel(nn.Module):
    """
    A score model which combines the scores from multiple models.

    This score model class allows for multiple score models to combine their
    scores arbitrarily. They may share all, some, or none of the model space
    with this class handling the bookkeeping. The scores from each model (where
    they use the same model dimensions) are simply summed. The class may also
    handle multiple inputs, internally they are combined into a single massive
    concatenated ``x`` vector, when passed to the models the ``x`` vector is
    split into the appropriate segments ``x_0, x_1, ..., x_n`` and each one is
    converted into the expected shape (defined by the ``x_shapes`` argument).

    Usage: a list of M models is passed to the constructor, these models will be
    used to compute the score. The x vector is split into N segments defined by
    the x_shapes argument (N does not need to equal M). The model_uses argument
    identifies which segments of x (defined by x_shapes) each model uses. For
    example Imagine three models [M1, M2, M3], and x_shapes = [(2, 3), (3, 4),
    (4, 5)], and model_uses = [[0, 1], [0, 2], None]. This means that M1 uses
    the first two segments of x (M1(x1, x2)), M2 uses the first and third
    segments of x (M2(x1, x3)), and M3 uses the full x vector (as a flat tensor
    M3(x)). The score will be broken up into similar segments and summed then
    returned as a flat tensor like x.

    Args:
        sde: The SDE that the score model is associated with. models: A list of
        score models. x_shapes: A list of shapes for the x vectors that the
        models expect.
            These are the shapes that the flat-concatenated ``x`` vector will be
            split into.
        model_uses: A list of lists of integers, where each list is the indices
            of the x vectors corresponding to ``x_shapes`` that each model uses.
            If None, the model will be passed the full ``x`` vector.
    """

    def __init__(
        self,
        sde: SDE,
        models: List[ScoreModel],
        x_shapes: Tuple[Tuple[int]],
        model_uses: Tuple[Union[None, Tuple[int]]],
        **kwargs
    ):
        super().__init__()
        self.sde = sde
        self.models = models
        self.x_shapes = x_shapes
        self.model_uses = model_uses

    def split_x(self, x: Tensor):
        B, D = x.shape

        # Split x into segments
        sub_x = []
        place = 0
        for shapex in self.x_shapes:
            sub_x.append(x[..., place : place + np.prod(shapex)].reshape(B, *shapex))
            place += np.prod(shapex)
        assert place == D
        return sub_x

    def join_x(self, sub_x: List[Tensor]):
        B = sub_x[0].shape[0]
        return torch.cat(tuple(S.reshape(B, -1) for S in sub_x), dim=-1)

    @property
    def xsize(self):
        return sum(np.prod(shapex) for shapex in self.x_shapes)

    def forward(self, t: Tensor, x: Tensor, *args, **kwargs):
        # Split x into segments
        sub_x = self.split_x(x)

        # Compute / store the score for each model
        scores = list(torch.zeros_like(sx) for sx in sub_x)
        for i, model in enumerate(self.models):
            # Select the segments from x that this model uses
            if self.model_uses[i] is None:
                model_x = x
                model_score = model(t, model_x, *args, **kwargs)
            else:
                model_x = tuple(sub_x[j] for j in self.model_uses[i])
                # Compute the score for this model
                model_score = model(t, *model_x, *args, **kwargs)

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
        return js * self.sde.sigma(t).view(-1, *[1] * len(js.shape[1:]))
