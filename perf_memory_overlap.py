"""
Performance for memory overlap.
"""

# Copyright (c) 2017, Gilles Legoux <gilles.legoux@gmail.com>
# All rights reserved.
# License: BSD 3 clause see https://choosealicense.com/licenses/bsd-3-clause/

import sys
import os
from os.path import exists, basename, splitext

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from memory_overlap import share_memory
from perf import timeit


def high_dim(n):
    x = np.arange(5 ** n)
    return x.reshape(*([5] * n))[[slice(0, 5 ** n, 2)] * n], \
           x.reshape(*([5] * n))[[slice(1, 5 ** n, 2)] * n]


def odd_even(n):
    x = np.arange(n)
    return x[1::2], x[::2]


def range_perf(func, n_min=1, max_n=100, step=1, get_arrays=odd_even,
               title_axis='n_elts',
               unity='ms', output=None):
    if exists(output):
        raise IOError('{} exists already!'.format(output))
    func_perf = timeit(show_only_time=True,
                       show_without_unit=True,
                       unity=unity)(func)
    if output is not None:
        stdout = sys.stdout
        sys.stdout = open(output, 'a+')
    print('{},time ({})'.format(title_axis, unity))
    for n in range(n_min, max_n + 1, step):
        print('{},'.format(n), end='')
        func_perf(*get_arrays(n))
    if output is not None:
        sys.stdout.close()
        sys.stdout = stdout


def trace_perf(pathname, title=''):
    df = pd.read_csv(pathname, index_col=0)
    df.plot(title=title)
    img_name, _ = splitext(basename(pathname))
    plt.savefig(img_name + '.jpg')


if __name__ == "__main__":
    print('=== Performance ===')

    naive_share_memory = timeit()(share_memory)
    np_shares_memory = timeit()(np.shares_memory)

    print('perf_01:')
    x = np.arange(6)
    a = x[::2]
    b = x[1::2]
    naive_share_memory(a, b)
    np_shares_memory(a, b)

    print('\nperf_02:')
    x = np.arange(7 * 8, dtype=np.int8).reshape(7, 8)
    a = x[::2, ::3]
    b = x[1::5, ::2]
    naive_share_memory(a, b)
    np_shares_memory(a, b)

    print('\nperf_03:')
    x = np.arange(4 * 20).reshape(4, 20).astype(np.int8)
    a = x[:, ::7]
    b = x[:, 3::3]
    naive_share_memory(a, b)
    np_shares_memory(a, b)

    print('\nperf_04:')
    x = np.arange(7 * 8 * 4, dtype=np.int32)
    np.random.shuffle(x)
    x = x.reshape(7, 8, 4)
    a = x[::, 1::3, ::]
    b = x[1::, 3::4, ::]
    naive_share_memory(a, b)
    np_shares_memory(a, b)
    print()

    # variant n_elts
    range_perf(share_memory, max_n=10 ** 6, step=10 ** 5, unity='ms',
               output='naive_perf_n_elts.csv')
    trace_perf('naive_perf_n_elts.csv')
    range_perf(np.shares_memory, max_n=10 ** 6, step=10 ** 5, unity='ns',
               output='np_perf_n_elts.csv')
    trace_perf('np_perf_n_elts.csv')

    # variant n_dims
    range_perf(share_memory, max_n=12, step=1, get_arrays=high_dim,
               title_axis='n_dims', unity='ms',
               output='naive_perf_n_dims.csv')
    trace_perf('naive_perf_n_dims.csv')
    range_perf(np.shares_memory, max_n=12, step=1, get_arrays=high_dim,
               title_axis='n_dims', unity='ns',
               output='np_perf_n_dims.csv')
    trace_perf('np_perf_n_dims.csv')
