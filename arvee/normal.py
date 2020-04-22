import numpy as np

from .random_variable import RandomVariable
from .uniform import Uniform


class StandardNormal(RandomVariable):
    def __init__(self, mu: float = 0, sigma2: float = 1, 
                 i: int = 100, n: int = 500):
        self._i = i
        self._n = n

    def generate(self, size: int = 1):
        V = Uniform()

        gen_size = size if size > self._n else self._n

        S = np.array([V.generate(self._i) for _ in range(gen_size)])
        S = S.sum(axis=1)

        return ((S - S.mean()) / np.sqrt(S.var()))[:size]
    
    def __call__(self, size: int = 1):
        for x in self.generate():
            yield x


class Normal(RandomVariable):
    _hist_size = 1000
    _hist_bins = 100

    def __init__(self, mu: float = 0, sigma: float = 1, 
                 standard_normal: StandardNormal = StandardNormal()):
        self.mu = mu
        self.sigma = sigma

        self._standard_normal = standard_normal

    def generate(self, size: int = 1):
        return self.mu + self.sigma * self._standard_normal.generate(size)
    
    def __call__(self, size: int = 1):
        for x in self.generate(size):
            yield x

    def _density_support(self):
        return np.linspace(-self.sigma ** 2, self.sigma ** 2, 100)

    def density(self, support: np.array = None):
        support = self._density_support()
        density = 1 / (self.sigma * np.sqrt(2 * np.math.pi)) * \
            np.exp(- 0.5 * ((support - self.mu) / self.sigma) ** 2)

        return support, density