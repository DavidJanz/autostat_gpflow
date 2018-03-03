import numpy as np
from itertools import chain
import copy


class AbstractKernelBaseClass:
    @property
    def kernels(self):
        raise NotImplemented()

    def simplify(self):
        raise NotImplemented()

    def _make_canonic(self):
        raise NotImplemented()

    @property
    def gpf_kernel(self):
        raise NotImplemented()

    def clone(self):
        return copy.deepcopy(self)


class KernelWrapper(AbstractKernelBaseClass):
    def __init__(self, kernel):
        kernel.parent = self
        self.kernel = kernel

    def __repr__(self):
        return str("Wrapped Kernel {}".format(self.root))

    @property
    def kernels(self):
        return [k for k in self.kernel]

    def add_child(self, child):
        self.kernel = child
        child.parent = self

    def rem_child(self, child):
        if self.kernel == child:
            self.kernel = None
            child.parent = None

    def simplify(self):
        if self.kernel.is_operator and len(self.kernel.children) == 1:
            self.kernel = self.kernel.children[0]
        self.kernel.simplify()
        self._make_canonic()

    def _make_canonic(self):
        pass

    @property
    def gpf_kernel(self):
        return self.kernel.gpf_kernel


class AbstractKernel(AbstractKernelBaseClass):
    def __init__(self, name):
        self.name = name
        self.parent = None
        self._children = []

        self._params_fixed = False

    def __repr__(self):
        raise NotImplementedError('Must override')

    def __iter__(self):
        for child in chain(*map(iter, self._children)):
            yield child
        yield self

    @property
    def kernels(self):
        return [k for k in self]

    def simplify(self):
        raise NotImplemented()

    def make_canonic(self):
        self._children = sorted(self._children, key=lambda child: child.name)

    @property
    def gpf_kernel(self):
        raise NotImplemented()

    @property
    def is_operator(self):
        raise NotImplemented()

    @property
    def is_toplevel(self):
        return isinstance(self.parent, KernelWrapper)

    def fix_parameters(self):
        raise NotImplemented()

    def __ensure_consistent(self):
        if not self.is_operator:
            self._params = self.params

    def __deepcopy__(self, memo):
        self.__ensure_consistent()
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == '_anchored_gpf_kern':
                setattr(result, k, None)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


class OperatorKernel(AbstractKernel):
    def __init__(self, name, kernels):
        super().__init__(name)
        for k in kernels:
            self.add_child(k)

    def __repr__(self):
        return "({} {})".format(self.name, " ".join(map(str, self.children)))

    @property
    def children(self):
        return self._children

    def add_child(self, child):
        self._children.append(child)
        child.parent = self

    def rem_child(self, child):
        self._children.remove(child)
        child.parent = None

    def simplify(self):
        for child in self.children:
            child.simplify()
            if child.is_operator and child.name == self.name:
                for grandkid in child.children:
                    self.add_child(grandkid)
                self.rem_child(child)
            if child.is_operator and len(child.children) == 1:
                self.rem_child(child)
                self.add_child(child.children[0])
            child.make_canonic()

    @property
    def is_operator(self):
        return True

    def fix_parameters(self):
        for child in self.children:
            child.fix_parameters()


class BaseKernel(AbstractKernel):
    def __init__(self, name, params):
        self._n_params = None
        self._gpf_kern_method = None
        self._anchored_gpf_kern = None

        super().__init__(name)

        if params is not None and len(params) != self._n_params:
            raise ValueError("Expected {} parameters, got {}"
                             .format(self._n_params, len(params)))

        self._params = params

    def __repr__(self):
        return self.name

    @property
    def is_operator(self):
        return False

    def simplify(self):
        pass

    @property
    def params(self):
        params = list(np.array([p.read_value() for p
                                in self.gpf_kernel.parameters]))
        return params

    @params.setter
    def params(self, params):
        for gpf_param, param in zip(self.gpf_kernel.parameters, params):
            gpf_param.assign(np.array(param))

    def fix_parameters(self):
        self._params_fixed = True

    @property
    def is_fixed(self):
        return self._params_fixed

    @property
    def is_anchored(self):
        return self._anchored_gpf_kern is not None

    @property
    def gpf_kernel(self):
        if not self.is_anchored:
            self._anchored_gpf_kern = self._gpf_kern_method(1)
            if self._params is not None:
                self.params = self._params

        return self._anchored_gpf_kern
