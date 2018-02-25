import numpy as np
import gpflow as gpf
import kernels.kernels_abstract


# OPERATOR KERNELS
class SumKernel(kernels.kernels_abstract.OperatorKernel):
    def __init__(self):
        super().__init__('+')

    @property
    def gpf_kernel(self):
        return gpf.kernels.Add([c.gpf_kernel for c in self.children])


class ProdKernel(kernels.kernels_abstract.OperatorKernel):
    def __init__(self):
        super().__init__('*')

    @property
    def gpf_kernel(self):
        return gpf.kernels.Prod([c.gpf_kernel for c in self.children])


# BASE KERNELS
class SEKernel(kernels.kernels_abstract.BaseKernel):
    def __init__(self, params=None):
        self._n_params = 2
        super().__init__('SE', params)
        self._gpf_kern_method = gpf.kernels.RBF


class PerKernel(kernels.kernels_abstract.BaseKernel):
    def __init__(self, params=None):
        self._n_params = 3
        super().__init__('PER', params)
        self._gpf_kern_method = gpf.kernels.PeriodicKernel


class LinKernel(kernels.kernels_abstract.BaseKernel):
    def __init__(self, params=None):
        self._n_params = 1
        super().__init__('LIN', params)
        self._gpf_kern_method = gpf.kernels.Linear
