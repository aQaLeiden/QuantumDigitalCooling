"""Tests for spinmodels.py."""
import unittest

from qdclib.spinmodels import *

class TFIMChain_Test(unittest.TestCase):

    def test_ground_state_dense(self):
        system = TFIMChain(3, 2, 1)
        self.assertAlmostEqual(
            system.ground_space_fidelity(
                system.ground_space_projector(normalized=True)),
            1)
        system = TFIMChain(3, 1, 1)
        self.assertAlmostEqual(
            system.ground_space_fidelity(
                system.ground_space_projector(normalized=True)),
            1)
        system = TFIMChain(3, 0.5, 1)
        self.assertAlmostEqual(
            system.ground_space_fidelity(
                system.ground_space_projector(normalized=True)),
            1)

    def test_ground_state_sparse(self):
        system = TFIMChain(3, 2, 1, sparse=True)
        self.assertAlmostEqual(
            system.ground_space_fidelity(
                system.ground_space_projector(normalized=True)),
            1)
        system = TFIMChain(3, 1, 1, sparse=True)
        self.assertAlmostEqual(
            system.ground_space_fidelity(
                system.ground_space_projector(normalized=True)),
            1)
        system = TFIMChain(3, 0.5, 1, sparse=True)
        self.assertAlmostEqual(
            system.ground_space_fidelity(
                system.ground_space_projector(normalized=True)),
            1)
