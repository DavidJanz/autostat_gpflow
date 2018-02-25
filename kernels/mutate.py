from kernels import kernels
import copy


_kern_base = [kernels.LinKernel, kernels.SEKernel, kernels.PerKernel]
_kern_op = [kernels.SumKernel, kernels.ProdKernel]


def mutate(root):
    for i, root_k in enumerate(root.kernels):
        if root_k.is_operator:
            continue

        # REPLACE
        for base in _kern_base:
            root_copy = copy.deepcopy(root)
            k = root_copy.kernels[i]
            if k.name != base:
                k.replace(base())
            else:
                continue
            root_copy.simplify()
            yield root_copy

        # EXPAND
        for op in _kern_op:
            for base in _kern_base:
                root_copy = copy.deepcopy(root)
                k = root_copy.kernels[i]
                new_parent = op()
                new_parent.add_child(base())
                k = k.extend(new_parent)
                root_copy.simplify()
                yield root_copy

        # REMOVE
        root_copy = copy.deepcopy(root)
        k = root_copy.kernels[i]
        if k.is_toplevel:
            continue
        k.remove()
        root_copy.simplify()
        yield root_copy
