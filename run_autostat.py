#!/usr/bin/env python3
import numpy as np

from kernels import kernels_abstract, kernel_defs, mutate
import gpflow as gpf
from time import time


def test_kernel(root, x, y):
    m = gpf.models.GPR(x, y, kern=root.root.gpf_kernel)
    m.likelihood.variance = 0.001

    t0 = time()
    gpf.train.ScipyOptimizer().minimize(m)
    print(time() - t0)

    ll = m.likelihood_tensor.eval(session=m.enquire_session())
    return root, ll


def center(arr):
    arr -= np.mean(arr)
    arr /= np.std(arr)
    return arr


k1 = kernel_defs.SEKernel()
k = kernels_abstract.KernStruct(k1)

seen = []

data = np.load("data/co2.npz")
x, y = center(data['x']).reshape(-1, 1), center(data['y']).reshape(-1, 1)

for m in mutate.mutatation_generator(k):
    seen.append(test_kernel(m, x, y))

seen = sorted(seen, key=lambda x: x[-1])
top_kernel, top_ll = seen[0]