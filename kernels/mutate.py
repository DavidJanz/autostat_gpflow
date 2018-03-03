import os
import itertools

from kernels import kernel_defs


_kern_base = [kernel_defs.LinKernel, kernel_defs.SEKernel, kernel_defs.PerKernel]
_kern_op = [kernel_defs.SumKernel, kernel_defs.ProdKernel]
_ext_pairs = list(itertools.product(_kern_base, _kern_op))


def replace(old_kernel, new_kernel):
    parent = old_kernel.parent
    parent.rem_child(old_kernel)
    parent.add_child(new_kernel)


def extend(old_kernel, new_kernel, op_rule):
    parent = old_kernel.parent
    parent.rem_child(old_kernel)
    parent.add_child(op_rule([old_kernel, new_kernel]))


def remove(old_kernel):
    parent = old_kernel.parent
    parent.rem_child(old_kernel)


def mutatation_generator(root):
    yield root.clone()
    for i, root_k in enumerate(root.kernels):
        if root_k.is_operator:
            continue

        # REPLACE
        for base_rule in _kern_base:
            root_copy = root.clone()
            replacement = base_rule()
            if root_copy.kernels[i].name != replacement.name:
                replace(root_copy.kernels[i], replacement)
            else:
                continue
            yield root_copy

        # EXPAND
        for (base_rule, op_rule) in _ext_pairs:
            root_copy = root.clone()
            extend(root_copy.kernels[i], base_rule(), op_rule)
            yield root_copy

        # REMOVE
        root_copy = root.clone()
        if root_copy.kernels[i].is_toplevel:
            continue
        remove(root_copy.kernels[i])
        yield root_copy
