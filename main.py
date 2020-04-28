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

    LN = LogNormal(1, 4)
    fig = LN.histogram()
    x, y = LN.density()
    plt.plot(x, y)
    plt.title('LogNormal(1, 4)')
    fig.savefig('plots/lognormal14.png')
    plt.close(fig)
    print('==== LogNormal(1, 4) ====')
    print(f'Mean:\t\t{np.mean(LN.generate(1000)):.2f} | 1')
    print(f'Variance:\t{np.var(LN.generate(1000)):.2f} | 4')

    LN = LogNormal(5, 2)
    fig = LN.histogram()
    x, y = LN.density()
    plt.plot(x, y)
    plt.title('LogNormal(5, 2)')
    fig.savefig('plots/lognormal52.png')
    plt.close(fig)
    print('==== LogNormal(5, 2) ====')
    print(f'Mean:\t\t{np.mean(LN.generate(1000)):.2f} | 5')
    print(f'Variance:\t{np.var(LN.generate(1000)):.2f} | 2')

    fig = plt.figure()
    x = np.log(LN.generate(1000))
    plt.hist(np.log(LN.generate(1000)), 30, density=True)
    fig.savefig('plots/log_lognormal.png')
    plt.title('log(LogNormal(1, 1))')
    plt.close(fig)
    print('==== log(LogNormal(1, 1) ====')
    print(f'Mean:\t\t{np.mean(x):.2f} | {LN.mu:.2f}')
    print(f'Variance:\t{np.var(x):.2f} | {LN.sigma2:.2f}')

    B = Binomial(5, 0.8)
    fig = B.histogram()
    x, y = B.density()
    plt.plot(x, y)
    plt.title('Binomial(5, 0.8)')
    fig.savefig('plots/binomial50_8.png')
    plt.close(fig)
    print('==== Binomial(5, 0.8) ====')
    print(f'Mean:\t\t{np.mean(B.generate(1000)):.2f} | 4')
    print(f'Variance:\t{np.var(B.generate(1000)):.2f} | 0.8')

    B = Binomial(50, 0.4)
    fig = B.histogram()
    x, y = B.density()
    plt.plot(x, y)
    plt.title('Binomial(50, 0.4)')
    fig.savefig('plots/binomial500_4.png')
    plt.close(fig)
    print('==== Binomial(50, 0.4) ====')
    print(f'Mean:\t\t{np.mean(B.generate(1000)):.2f} | 20')
    print(f'Variance:\t{np.var(B.generate(1000)):.2f} | 12')