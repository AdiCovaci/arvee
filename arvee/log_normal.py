import numpy as np

from .random_variable import RandomVariable
from .normal import Normal


class LogNormal(RandomVariable):
    _hist_size = 1000
    _hist_bins = 100

    def __init__(self, m: float = 0, s2: float = 1):
        self.m = m
        self.s = np.sqrt(s2)
        self.s2 = s2

        log_sm = np.log(s2 / (m ** 2) + 1)
        self.mu = np.log(m) - 0.5 * log_sm
        self.sigma2 = log_sm
        self.sigma = np.sqrt(self.sigma2)
    
    def generate(self, size: int = 1):
        X = Normal(self.mu, self.sigma)
        return np.exp(X.generate(size))
    
    def __call__(self, size: int = 1):
        for x in self.generate(size):
            yield x

    def _density_support(self):
        return np.linspace(0.01, 30, 100)

    def density(self, support: np.array = None):
        support = self._density_support()
        density = 1 / (self.sigma * support * np.sqrt(2 * np.math.pi)) * \
            np.exp(- 0.5 * ((np.log(support) - self.mu) / self.sigma) ** 2)

        return support, density
