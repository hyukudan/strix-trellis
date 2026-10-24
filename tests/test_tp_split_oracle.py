import importlib.util
from pathlib import Path
import unittest
from fractions import Fraction
import numpy as np

spec = importlib.util.spec_from_file_location('split_oracle', Path(__file__).resolve().parents[1] / 'tools/tp_split_oracle.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


class SplitOracle(unittest.TestCase):
    def test_rounding_ties(self):
        self.assertEqual(oracle.round32(Fraction(1) + Fraction(1,2**24)), np.float32(1))
        self.assertEqual(oracle.round32(Fraction(1) + Fraction(3,2**24)).view('u4'), 0x3f800002)
        self.assertEqual(oracle.round32(Fraction(1,2**150)).view('u4'), 0)
        self.assertEqual(oracle.round32(-Fraction(1,2**150)).view('u4'), 0x80000000)

    def test_ideal_dot2_rounding_hypothesis(self):
        # This hypothesis is a precision reference, not an ISA bit emulator.
        self.assertEqual(oracle.fdot2(1,1,2**-12,2**-12,-1), np.float32(2**-24))
        self.assertEqual(oracle.fdot2(1,1,1,1,0), np.float32(2))
        self.assertEqual(oracle.fdot2(-0.,1.,-0.,1.,-0.).view('u4'),0x80000000)
        self.assertEqual(oracle.fdot2(1,1,2**-12,2**-12,2**-24).view('u4'),0x3f800001)
        self.assertEqual(oracle.fdot2(2**-12,2**-12,1,1,2**-24).view('u4'),0x3f800001)
        self.assertEqual(oracle.round32(Fraction(1)+Fraction(1,2**23)).view('u4'),0x3f800001)
        self.assertEqual(np.float32(np.float32(1+2**-24)+np.float32(2**-24)).view('u4'),0x3f800000)

    def test_split_order_and_arrival(self):
        rng = np.random.default_rng(901)
        q = rng.normal(size=(128,3)).astype('f2')
        a = rng.normal(size=(2,128)).astype('f2')
        for splits in (4,16,32):
            p = oracle.partials(q,a,splits)
            expected = oracle.fold(p)
            for _ in range(5):
                self.assertEqual(oracle.fold(p,rng.permutation(splits)).tobytes(), expected.tobytes())
            for row in range(2):
                self.assertEqual(oracle.partials(q,a[row:row+1],splits).tobytes(),p[:,row:row+1].tobytes())
        self.assertNotEqual(oracle.fold(oracle.partials(q,a,4)).tobytes(),oracle.fold(oracle.partials(q,a,16)).tobytes())


if __name__ == '__main__':
    unittest.main()
