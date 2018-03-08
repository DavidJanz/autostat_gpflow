import gpflow as gpf
from . import kernels_abstract


# OPERATOR KERNELS
class SumKernel(kernels_abstract.OperatorKernel):
    def __init__(self, kernels):
        super().__init__('+', kernels)

    @property
    def gpf_kernel(self):
        return gpf.kernels.Sum([c.gpf_kernel for c in self.children])


class ProdKernel(kernels_abstract.OperatorKernel):
    def __init__(self, kernels):
        super().__init__('*', kernels)

    @property
    def gpf_kernel(self):
        return gpf.kernels.Product([c.gpf_kernel for c in self.children])


# BASE KERNELS
class SEKernel(kernels_abstract.BaseKernel):
    def __init__(self, params=None):
        self._n_params = 2
        super().__init__('SE', params)
        self._gpf_kern_method = gpf.kernels.RBF


class PerKernel(kernels_abstract.BaseKernel):
    """Note, this is not the same periodic kernel as in autostat.
    Periodic as defined in autostat paper has to be implemented in gpflow."""
    def __init__(self, params=None):
        self._n_params = 3
        super().__init__('PER', params)
        self._gpf_kern_method = gpf.kernels.Periodic


class LinKernel(kernels_abstract.BaseKernel):
    def __init__(self, params=None):
        self._n_params = 1
        super().__init__('LIN', params)
        self._gpf_kern_method = gpf.kernels.Linear
