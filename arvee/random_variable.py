import matplotlib.pyplot as plt
import numpy as np


class RandomVariable:
    _hist_size = None
    _hist_bins = None

    def __init__(self):
        raise NotImplementedError

    def _generate(self):
        raise NotImplementedError

    def __call__(self, size: int = 1):
        for _ in range(size):
            yield self._generate()

    def generate(self, size: int = 1):
        return np.array([x for x in self(size)])

    def density(self):
        raise NotImplementedError

    def _density_support(self):
        raise NotImplementedError

    def histogram(self):
        fig = plt.figure()
        plt.hist(self.generate(self._hist_size), self._hist_bins, density=True)
        return fig
