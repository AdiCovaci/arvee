#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os

from arvee import RandomVariable, Bernoulli, Binomial, Uniform, Normal, LogNormal

if __name__ == '__main__':
    try:
        os.mkdir('plots')
    except OSError:
        pass

    X = LogNormal(1, 4)
    fig = X.histogram()
    x, y = X.density()
    plt.plot(x, y)
    fig.savefig('plots/lognormal.png')
    plt.title('LogNormal(1, 4)')
    plt.close(fig)




    # B = Binomial(20, 0.2)
    # print(B.generate())

    # N = Normal()
    # fig = histogram(N)
    # fig.savefig('plots/normal.png')
    # plt.title('Normal(0, 1)')
    # plt.close(fig)

    # LN = LogNormal(1, 1)
    # fig = histogram(LN)
    # fig.savefig('plots/lognormal.png')
    # plt.title('LogNormal(1, 1)')
    # plt.close(fig)

    # fig = plt.figure()
    # plt.hist(np.log(LN.generate(100)), 30, density=True)
    # fig.savefig('plots/log_lognormal.png')
    # plt.title('log(LogNormal(1, 1))')
    # plt.close(fig)

