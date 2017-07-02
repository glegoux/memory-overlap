"""
Performance module.
"""

# Copyright (c) 2017, Gilles Legoux <gilles.legoux@gmail.com>
# All rights reserved.
# License: BSD 3 clause see https://choosealicense.com/licenses/bsd-3-clause/

import time
from functools import wraps


def timer(start, end, show_without_unit, unity=None):
    d = h = m = s = ms = us = ns = 0
    r = end - start
    if unity == 'd' or unity is None:
        d, r = divmod(end - start, 60 * 60 * 24)
    if unity == 'h' or unity is None:
        h, r = divmod(r, 60 * 60)
    if unity == 'm' or unity is None:
        m, r = divmod(r, 60)
    if unity == 's' or unity is None:
        s, r = divmod(r, 1)
    if unity == 'ms' or unity is None:
        ms, r = divmod(r * 10 ** 3, 1)
    if unity == 'us':
        unity = 'μs'
        us, r = divmod(r * 10 ** 6, 1)
    if unity is None:
        us, r = divmod(r * 10 ** 3, 1)
    if unity == 'ns':
        ns, r = divmod(r * 10 ** 9, 1)
    if unity is None:
        ns, r = divmod(r * 10 ** 3, 1)
    times = [int(x) for x in [d, h, m, s, ms, us, ns]]
    labels = ['d', 'h', 'min', 's', 'ms', 'μs', 'ns']
    if unity is None:
        ptimes = [x for x in zip(times, labels) if x[0] > 0]
    else:
        ptimes = [x for x in zip(times, labels) if x[1] == unity]
    res = ''
    for ptime in ptimes:
        if show_without_unit:
            res += '{} '.format(ptime[0])
        else:
            res += '{} {} '.format(*ptime)
    return res


def timeit(show_only_time=False, show_without_unit=False, unity=None):
    def wrapper_outer(func):
        @wraps(func)
        def wrapper_inner(*args_, **kwargs_):
            start = time.time()
            result = func(*args_, **kwargs_)
            end = time.time()
            if show_only_time:
                print('{}'.format(timer(start, end, show_without_unit, unity)))
            else:
                print('{}.{}: {}'.format(func.__module__, func.__name__,
                                         timer(start, end, unity)))
            return result

        return wrapper_inner

    return wrapper_outer
