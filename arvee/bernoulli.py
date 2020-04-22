import numpy as np

from .random_variable import RandomVariable
from .uniform import Uniform


class Bernoulli(RandomVariable):
    _hist_size = 50
    _hist_bins = 2

    def __init__(self, p: float = 0.5):
        self.p = p

    def generate(self, size: int = 1):
        X = Uniform()

        x = X.generate(size)
        x = (x < self.p).astype(int)
        return x
    
    def __call__(self, size: int = 1):
        for x in self.generate():
            yield x

    def _density_support(self):
        return np.linspace(0, 1, 100)

    def density(self, support: np.array = None):
        support = self._density_support()
        density = np.array([1 - self.p] * 50 + [self.p] * 50) * 2

        return support, density