import numpy as np

from .random_variable import RandomVariable


class Uniform(RandomVariable):
    _hist_size = 1000
    _hist_bins = 25

    def __init__(self, a: float = 0, b: float = 1):
        self.a = a
        self.b = b
    
    def _generate(self):
        return np.random.uniform(self.a, self.b)

    def generate(self, size: int = 1):
        return np.random.uniform(self.a, self.b, size)

    def _density_support(self):
        return np.linspace(self.a, self.b, 100)

    def density(self, support: np.array = None):
        support = self._density_support()
        density = np.repeat(1 / (self.b - self.a), support.shape[0])

        return support, density