import numpy as np

from .random_variable import RandomVariable
from .bernoulli import Bernoulli
from .normal import StandardNormal


class _SmallBinomial(RandomVariable):
    def __init__(self, n: int, p: float = 0.5):
        self.n = n
        self.p = p

        self._bernoulli = Bernoulli(p)

    def _generate(self):
        x = self._bernoulli.generate(self.n)
        return np.sum(x)


class _LargeBinomial(RandomVariable):
    def __init__(self, n: int, p: float = 0.5, 
                 standard_normal: StandardNormal = StandardNormal()):
        self.n = n
        self.p = p

        self._standard_normal = standard_normal

    def generate(self, size: int = 1):
        w = self._standard_normal.generate(size)
        return np.floor(self.n * self.p + \
            w * np.sqrt(self.n * self.p * (1 - self.p)))
    
    def __call__(self, size: int = 1):
        for x in self.generate():
            yield x


class Binomial(RandomVariable):
    _hist_size = 1000

    def __init__(self, n: int, p: float = 0.5):
        if n * p >= 10 and n * (1 - p) >= 10:
            self._binomial = _LargeBinomial(n, p)
        else:
            self._binomial = _SmallBinomial(n, p)

        self._hist_bins = n

    def generate(self, size: int = 1):
        return self._binomial.generate(size)

    def __call__(self, size: int = 1):
        return self._binomial(size)

    def _density_support(self):
        return np.arange(self._binomial.n + 1)

    def density(self, support: np.array = None):
        support = self._density_support()

        n = self._binomial.n
        p = self._binomial.p

        density = np.array([np.math.factorial(n) / (np.math.factorial(k) * \
                   np.math.factorial(n - k)) * p ** k * (1 - p) ** (n - k) 
            for k in support])

        return support, density