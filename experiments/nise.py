import numpy as np

from moopt import nise
from moopt.scalarization_interface import scalar_interface, single_interface, \
    w_interface

class NISE:
    """NISE multi-objective algorithm.
    
    Find a Pareto-frontier of loss functions (original and contrastive).
    """
    def __init__(self, model, dataset, n_models):
        self.model = model
        self.dataset = dataset
        self.n_models = n_models
        self.optimize()

    def optimize(self):
        """Optimize NISE."""
        self.opt = nise(
            weightedScalar=Scalarization(self.model, self.dataset),
            singleScalar=Scalarization(self.model, self.dataset),
            targetSize=self.n_models,
            norm=True,
        )
        self.opt.optimize()

class Scalarization(w_interface, single_interface, scalar_interface):
    """Scalarization interface for NISE.
    
    Given a model and a dataset, it is able to receive weights, train the
    model, and return its losses.
    """
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.__M = 2

    @property
    def M(self):
        return self.__M

    @property
    def feasible(self):
        return True

    @property
    def optimum(self):
        return True

    @property
    def objs(self):
        return self.__objs

    @property
    def x(self):
        return self.__x

    @property
    def w(self):
        return self.__w

    def optimize(self, w):
        """Calculates the multiobjective scalarization."""
        if type(w) is int:
            self.__w = np.zeros(self.M)
            self.__w[w] = 1.0
            self.__w[w - 1] = 0.0
        elif type(w) is np.ndarray and w.ndim == 1 and w.size == self.M:
            self.__w = w
        else:
            raise('`w` is in the wrong format.')
        w1, w2 = self.__w     
        self.model.fit(self.dataset, w1, w2)
        self.__objs = self.model.losses(self.dataset)
        self.__x = self.model.copy()

        return self
