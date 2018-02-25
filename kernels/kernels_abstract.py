import numpy as np
from itertools import chain
import copy


class KernStruct:
    def __init__(self, kern):
        kern.parent = self
        self.root = kern
        self.seen = []

    def __repr__(self):
        return str(self.root)

    @property
    def kernels(self):
        return [k for k in self.root]

    def simplify(self):
        self.root.simplify()

    def add_child(self, child):
        self.root = child

    def rem_child(self, child):
        if self.root == child:
            self.root = None

    def make_canonic(self):
        self.root.make_canonic()


class AbstractKernelStructure:
    def __init__(self, name):
        self.name = name
        self._children = []
        self.parent = None

    def __repr__(self):
        raise NotImplementedError('Must override')

    def __iter__(self):
        for child in chain(*map(iter, self._children)):
            yield child
        yield self

    def simplify(self):
        pass

    @property
    def is_toplevel(self):
        return self.parent is None

    @property
    def gpf_kernel(self):
        raise NotImplementedError('Must override')

    def fix(self):
        if not self.is_operator:
            self._fixed = True
        for child in self._children:
            child.fix()

    def __ensure_consistent(self):
        if self.is_operator:
            self._params = self.params

    def __deepcopy__(self, memo):
        self.__ensure_consistent
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == '_anchored_gpf_kern':
                setattr(result, k, None)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def make_canonic(self):
        self._children = sorted(self._children,
                    key=lambda child: child.name)


class OperatorKernel(AbstractKernelStructure):
    def __init__(self, name):
        super().__init__(name)

    def __repr__(self):
        return "({} {})".format(
            self.name, " ".join(map(str, self.children)))

    @property
    def children(self):
        return self._children

    @property
    def is_lonely(self):
        return len(self.children) == 1

    def add_child(self, child):
        self._children.append(child)
        child.parent = self

    def rem_child(self, child):
        self._children.remove(child)

    def simplify(self):
        for child in self.children:
            child.simplify()
            if child.is_operator and child.name == self.name:
                grandkids = child.children
                for grandkid in child.children:
                    self.add_child(grandkid)
                self.rem_child(child)
        if self.is_toplevel and self.is_lonely:
            return self.children[0]
        if self.is_toplevel:
            return self

    @property
    def is_operator(self):
        return True

class BaseKernel(AbstractKernelStructure):
    def __init__(self, name, params):
        super().__init__(name)
        if params is not None and len(params) != self._n_params:
            raise ValueError("Expected {} parameters, got {}"
                    .format(self._n_params, len(params)))
        self._params = params
        self._gpf_kern_method = None
        self._anchored_gpf_kern = None
        self._fixed = False

    def __repr__(self):
        if self.is_operator:
            return "({} {})".format(
                self.name, " ".join(map(str, self.children)))
        return self.name

    def extend(self, new_parent):
        self.replace(new_parent)
        new_parent.add_child(self)

    def replace(self, new_child):
        if self.is_toplevel:
            self = new_child
        else:
            self.parent.add_child(new_child)
            self.remove()

    def remove(self):
        if self.is_toplevel:
            raise ValueError("Cannot remove toplevel kernel")
        self.parent.rem_child(self)

    @property
    def is_operator(self):
        return False

    @property
    def params(self):
        params = list(np.array([p.read_value() for
                    p in self.gpf_kernel.parameters]))
        return params

    @params.setter
    def params(self, params):
        for gpf_param, param in zip(self.gpf_kernel.parameters, params):
            gpf_param.assign(np.array(param))

    @property
    def is_fixed(self):
        return self._fixed

    @property
    def is_anchored(self):
        return not self._anchored_gpf_kern is None

    @property
    def gpf_kernel(self):
        if not self.is_anchored:
            self._anchored_gpf_kern = self._gpf_kern_method(1)
            if self._params is not None:
                self.params = self._params

        return self._anchored_gpf_kern
