"""Tests for spinmodels.py."""
import unittest

import numpy as np

from .qdcnumpy import (continuous_logsweep_protocol,)
from .spinmodels import TFIMChain

class qdcnumpy_Test(unittest.TestCase):

    def test_continuous_logsweep_dm(self):
        K = 3
        for L in [2, 3]:
            system = TFIMChain(L, 1, 1)
            init_state = np.eye(2**L, dtype=complex) / (2**L)
            state = continuous_logsweep_protocol(init_state, system, K)
            self.assertAlmostEqual(np.trace(state), 1)
            self.assertGreater(system.eigenstate_occupations(state)[0], 0.5)

    def test_continuous_logsweep_wf_single(self):
        K = 3
        for L in [2, 3]:
            system = TFIMChain(L, 1, 1)
            init_state = np.zeros(2**L, dtype=complex)
            init_state[0] = 1
            state = continuous_logsweep_protocol(init_state, system, K)
            self.assertEqual(init_state.shape, state.shape)
            self.assertAlmostEqual(np.sum(np.abs(state)**2), 1)

    def test_continuous_logsweep_wf_matr(self):
        K = 3
        for L in [2, 3]:
            system = TFIMChain(L, 1, 1)
            init_state = np.zeros((2**L, 2), dtype=complex)
            init_state[0, 0] = 1
            init_state[1, 1] = 1
            state = continuous_logsweep_protocol(init_state, system, K)
            self.assertEqual(init_state.shape, state.shape)
            for statevec in state.T:
                self.assertAlmostEqual(np.sum(np.abs(statevec)**2), 1)
