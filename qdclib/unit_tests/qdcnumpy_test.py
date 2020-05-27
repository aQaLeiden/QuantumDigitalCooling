"""Tests for spinmodels.py."""
import unittest

from qdclib.qdcnumpy import *
from qdclib.spinmodels import TFIMChain

class qdcnumpy_Test(unittest.TestCase):

    def test_continuous_logsweep(self):
        K = 3
        for L in [2, 3]:
            system = TFIMChain(L, 1, 1)
            init_state = np.eye(2**L, dtype=complex) / (2**L)
            state = continuous_logsweep_protocol(init_state, system, K)
            self.assertAlmostEqual(np.trace(state), 1)
            self.assertGreater(system.eigenstate_occupations(state)[0], 0.5)
