import itertools

import gpflow as gpf
from . import kernels_abstract
from . import periodic_nodc


# OPERATOR KERNELS
class SumKernel(kernels_abstract.OperatorKernel):
    def __init__(self, kernels):
        super().__init__('+', kernels)

    @property
    def gpf_kernel(self):
        return gpf.kernels.Sum([c.gpf_kernel for c in self.children])

    def break_into_summands(self):
        return [subop for op in self._children for subop in op.break_into_summands()]


class ProdKernel(kernels_abstract.OperatorKernel):
    def __init__(self, kernels):
        super().__init__('*', kernels)

    @property
    def gpf_kernel(self):
        return gpf.kernels.Product([c.gpf_kernel for c in self.children])

    def break_into_summands(self):
        # Recursively distribute each of the terms to be multiplied.
        distributed_ops = [op.break_into_summands() for op in self._children]

        # Now produce a sum of all combinations of terms in the products.
        new_prod_ks = [ProdKernel(kernels=prod) for prod in itertools.product(*distributed_ops)]
        return new_prod_ks


# BASE KERNELS
class SEKernel(kernels_abstract.BaseKernel):
    def __init__(self, params=None):
        super().__init__('SE')
        self._n_params = 2
        self._gpf_kern_method = gpf.kernels.RBF
        self.check_params(params)


class PerKernel(kernels_abstract.BaseKernel):
    """See definition in https://arxiv.org/pdf/1402.4304.pdf"""
    def __init__(self, params=None):
        super().__init__('PERNoDC')
        self._n_params = 3
        self._gpf_kern_method = periodic_nodc.PeriodicNoDC
        self.check_params(params)


class OriginalPerKernel(kernels_abstract.BaseKernel):
    """Note, this is not the same periodic kernel as in autostat.
    Periodic as defined in autostat paper has to be implemented in gpflow."""
    def __init__(self, params=None):
        super().__init__('PER')
        self._n_params = 3
        self._gpf_kern_method = gpf.kernels.Periodic
        self.check_params(params)


class LinKernel(kernels_abstract.BaseKernel):
    def __init__(self, params=None):
        super().__init__('LIN')
        self._n_params = 1
        self._gpf_kern_method = gpf.kernels.Linear
        self.check_params(params)
