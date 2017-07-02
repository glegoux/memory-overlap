"""
Testing for memory overlap.
"""

# Copyright (c) 2017, Gilles Legoux <gilles.legoux@gmail.com>
# All rights reserved.
# License: BSD 3 clause see https://choosealicense.com/licenses/bsd-3-clause/

from memory_overlap import share_memory
import numpy as np

import unittest


class TestMemoryOverlap(unittest.TestCase):
    def test_01(self):
        x = np.arange(6)
        a = x[::2]
        b = x[1::2]
        self.assertEqual(share_memory(a, b), False)

    def test_02(self):
        x = np.arange(7 * 8, dtype=np.int8).reshape(7, 8)
        a = x[::2, ::3]
        b = x[1::5, ::2]
        self.assertEqual(share_memory(a, b), True)

    def test_03(self):
        x = np.arange(4 * 20).reshape(4, 20).astype(np.int8)
        a = x[:, ::7]
        b = x[:, 3::3]
        self.assertEqual(share_memory(a, b), False)

    def test_04(self):
        x = np.arange(7 * 8 * 4, dtype=np.int32)
        np.random.shuffle(x)
        x = x.reshape(7, 8, 4)
        a = x[::, 1::3, ::]
        b = x[1::, 3::4, ::]
        self.assertEqual(share_memory(a, b), True)


if __name__ == '__main__':
    print('=== Test ===')
    unittest.main()
