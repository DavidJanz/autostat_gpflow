import numpy as np
from itertools import chain
import copy


class AbstractKernelBaseClass:
    """ABC for every Kernel-like object"""
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
    """The parent object for every kernel expression."""
    def __init__(self, kernel):
        kernel.parent = self
        self.kernel = kernel

    def __repr__(self):
        return str("Wrapped Kernel {}".format(self.kernel))

    @property
    def kernels(self):
        return [k for k in self.kernel]

    def add_child(self, child):
        """KernelWrapper has only one child at a time."""
        self.kernel = child
        child.parent = self

    def rem_child(self, child):
        """KernelWrapper has only one child at a time."""
        if self.kernel == child:
            self.kernel = None
            child.parent = None

    def simplify(self):
        # if kernel only has one child
        if self.kernel.is_operator and len(self.kernel.children) == 1:
            # skip kernel in hierarchy
            self.kernel = self.kernel.children[0]
        self.kernel.simplify()

    def _make_canonic(self):
        pass

    @property
    def gpf_kernel(self):
        return self.kernel.gpf_kernel


class AbstractKernel(AbstractKernelBaseClass):
    """Implements functionality shared between OperatorKernel and BaseKernel"""
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

    def _make_canonic(self):
        """Sorted children in alphabetical order of name to allow for
        easy equality comparisons between kernels."""
        self._children = sorted(self._children, key=lambda child: child.name)

    @property
    def gpf_kernel(self):
        raise NotImplemented()

    @property
    def is_operator(self):
        raise NotImplemented()

    @property
    def is_toplevel(self):
        """True if parent is a KernelWrapper"""
        return isinstance(self.parent, KernelWrapper)

    def fix_parameters(self):
        raise NotImplemented()

    def _ensure_consistent(self):
        """Synchronises gpf kernel and local data."""
        pass

    def __deepcopy__(self, memo):
        self._ensure_consistent()
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
    """Implements functionality for kernels that return a function of their
    children."""
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
        """Simplifies kernel expression."""
        for child in self.children:
            child.simplify()
            if child.is_operator:
                # if child if of same type as parent
                if child.name == self.name:
                    # collapse child into parent
                    for grandkid in child.children:
                        self.add_child(grandkid)
                    self.rem_child(child)
                # if child only has one child
                if len(child.children) == 1:
                    # skip the child in hierarchy
                    self.rem_child(child)
                    self.add_child(child.children[0])
        self._make_canonic()

    @property
    def is_operator(self):
        return True

    def fix_parameters(self):
        for child in self.children:
            child.fix_parameters()


class BaseKernel(AbstractKernel):
    """Implements functionality for individual base kernels."""
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
        """Reads parameters of the corresponding gpf kernel"""
        params = list(np.array([p.read_value() for p
                                in self.gpf_kernel.parameters]))
        return params

    @params.setter
    def params(self, params):
        """Sets parameters of the corresponding gpf kernel"""
        for gpf_param, param in zip(self.gpf_kernel.parameters, params):
            gpf_param.assign(np.array(param))

    def fix_parameters(self):
        self._params_fixed = True

    @property
    def is_fixed(self):
        return self._params_fixed

    @property
    def is_anchored(self):
        """Checks whether corresponding gpf kernel is initialised"""
        return self._anchored_gpf_kern is not None

    @property
    def gpf_kernel(self):
        """Returns corresponding gpf kernel. Creates if one is not anchored."""
        if not self.is_anchored:
            self._anchored_gpf_kern = self._gpf_kern_method(1)
            if self._params is not None:
                self.params = self._params

        return self._anchored_gpf_kern

    def _ensure_consistent(self):
        """Synchronises gpf kernel and local data. Params are copied to
        ensure __deepcopy__ has access to up-to-date parameters."""
        if not self.is_operator:
            self._params = self.params
