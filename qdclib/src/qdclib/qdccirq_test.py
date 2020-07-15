"""Tests for spinmodels.py."""
import unittest

import cirq

from .qdccirq import (bangbang_protocol,)
from .spinmodels import TFIMChain
from .qdcutils import trace_out

class qdccirq_Test(unittest.TestCase):

    def test_bangbang_protocol(self):
        system = TFIMChain(3, 1, 1, sparse=True)
        system.normalize()
        c = bangbang_protocol(system, ('Y'), 3)
        # expected circuit length:
        #    iterations * qubits * (len(HS) + len(coupl) + reset)
        self.assertEqual(len(c), 3 * 3 * (4 + 2 + 1))

        s = cirq.DensityMatrixSimulator()
        res = s.simulate(c)
        final_state = trace_out(res.final_density_matrix, -1)

        final_energy = system.energy_expval(final_state)
        self.assertLess(final_energy, 0)
