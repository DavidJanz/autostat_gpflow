import unittest
import os
import sys

PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJ_DIR not in sys.path:
    sys.path.append(PROJ_DIR)

import kernels.kernels_abstract as ka

# run tests with python -m unittest


class TestKernelWrapper(unittest.TestCase):
    def setUp(self):
        self.bk = ka.BaseKernel('test')
        self.kw = ka.KernelWrapper(self.bk)

    def test_add_child(self):
        old_child = self.kw.kernel
        new_child = ka.BaseKernel('new_child')
        self.kw.add_child(new_child)
        self.assertTrue(self.kw.kernel == new_child)
        self.assertTrue(new_child.parent == self.kw)
        self.assertTrue(old_child.parent is None)

    def test_rem_child(self):
        old_child = self.kw.kernel
        self.kw.rem_child(old_child)
        self.assertTrue(self.kw.kernel is None)
        self.assertTrue(old_child.parent is None)

    def test_simplify(self):
        simplified = ka.KernelWrapper(ka.BaseKernel('simplify'))
        self.kw.add_child(ka.OperatorKernel(
            name='simplify1',
            kernels=ka.OperatorKernel(
                name='simplify2', kernels=ka.BaseKernel('simplify'))))
        self.kw.simplify()
        self.assertEqual(str(self.kw), str(simplified))


if __name__ == '__main__':
    unittest.main()
