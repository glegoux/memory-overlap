"""
Solving memory overlap for two strided arrays `a` and `b` with naive method.

*Algorithm description*

Alternative solution to [1] (numpy library implementation) which resolves the
equivalent problen `bounded Diophantine equations with positive coefficients`.

Compute memory addresses of `a` then `b`, put that into two
`np.array` named respectively `a_addr` and `b_addr`, then compare if these
two matrices have common values (indices aren't important here).

*Complexity*

It is only a Python prototype, C implementation should be more efficient, but
it is not really performing.

Time complexity: O(a.size + b.size)
Space complexity: O(a.size + b.size)

References
----------

.. [1] https://github.com/numpy/numpy/blob/maintenance/1.11.x/numpy/core/src/private/mem_overlap.c
.. [2] https://en.wikipedia.org/wiki/Array_data_structure
.. [3] https://en.wikipedia.org/wiki/Stride_of_an_array
.. [4] https://en.wikipedia.org/wiki/Diophantine_equation
.. [5] https://en.wikipedia.org/wiki/NP-completeness
.. [6] https://fr.wikipedia.org/wiki/Endianness#Little_endian
.. [7] http://scipy-cookbook.readthedocs.io/items/ViewsVsCopies.html
.. [8] http://www.scipy-lectures.org/advanced/advanced_numpy/
"""

# Copyright (c) 2017, Gilles Legoux <gilles.legoux@gmail.com>
# All rights reserved.
# License: BSD 3 clause see https://choosealicense.com/licenses/bsd-3-clause/

from ctypes import string_at

import numpy as np


def get_addr(elt, *indices, base_addr, strides):
    indices = np.array(indices)
    return base_addr + np.sum(indices * strides)


def get_addrs(array):
    if array.size == 0:
        return np.array([], dtype=np.int64)
    base_addr = array[0].__array_interface__.get('data')[0]
    strides = np.array(array.strides)
    shapes = np.array(array.shape)
    indices = np.meshgrid(*[range(shape) for shape in shapes], indexing='ij')
    return np.vectorize(get_addr, excluded=['base_addr', 'strides'],
                        otypes=[np.int64]) \
        (array, *indices, base_addr=base_addr, strides=strides)


def get_hexs(addrs):
    return np.vectorize(hex)(addrs)


def get_value(addr, dtype):
    size = np.dtype(dtype).itemsize
    addr_content = string_at(int(addr), size)
    return np.fromstring(addr_content, dtype)[0]


def get_values(addrs, dtype):
    return np.vectorize(get_value, excluded=['dtype'], otypes=[dtype]) \
        (addrs, dtype=dtype)


def has_one_common_elt(array1, array2):
    set1 = set(array1.ravel())
    set2 = set(array2.ravel())
    return len(set1.intersection(set2)) > 0


def share_memory(x, y):
    x_addrs = get_addrs(x)
    y_addrs = get_addrs(y)
    return has_one_common_elt(x_addrs, y_addrs)


if __name__ == "__main__":
    # run in ipython console
    print('=== Example ===')
    dtype = np.int8
    array = np.arange(7 * 8, dtype=dtype)
    np.random.shuffle(array)
    array = array.reshape(7, 8)
    addrs = get_addrs(array)
    values = get_values(addrs, dtype)
    hexs = get_hexs(addrs)

    x = array
    a = array[::, 1::3]
    b = array[1::, 3::4]
