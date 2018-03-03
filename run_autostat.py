#!/usr/bin/env python3
import numpy as np
import os
from kernels import kernels_abstract, kernel_defs, mutate
import gpflow as gpf
import joblib
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def test_kernel(kernel_wrapper, x, y):
    print("testing kernel")
    m = gpf.models.GPR(x, y, kern=kernel_wrapper.gpf_kernel)
    print("building model")
    m.likelihood.variance = 0.001
    print("minimising")
    gpf.train.ScipyOptimizer().minimize(m)
    print("getting ll")
    ll = m.likelihood_tensor.eval(session=m.enquire_session())
    print("done testing kernel {}".format(ll))
    return ll


def center(arr):
    arr -= np.mean(arr)
    arr /= np.std(arr)
    return arr


n_steps = 6

results = []
seen = set()

data = np.load("data/co2.npz")
n_data = 25
x, y = center(data['x'][:n_data]).reshape(-1, 1), center(data['y'][:n_data]).reshape(-1, 1)

top_kernel = None
base_kernels = [kernels_abstract.KernelWrapper(kernel_defs.SEKernel()),
                kernels_abstract.KernelWrapper(kernel_defs.LinKernel()),
                kernels_abstract.KernelWrapper(kernel_defs.PerKernel())]

prospective_kernels = base_kernels

for step in range(n_steps):
    print("step {}".format(step))
    to_try = []
    for m in prospective_kernels:
        m.simplify()
        if m not in seen:
            to_try.append(m)
    seen.update(to_try)
    r = joblib.Parallel(n_jobs=2)(joblib.delayed(test_kernel)(m, x, y) for m in to_try)
    # r = [test_kernel(m, x, y) for m in to_try]
    results += zip(to_try, r)
    results = sorted(results, key=lambda x: x[-1], reverse=True)
    top_kernel, top_ll = results[0]

    prospective_kernels = mutate.mutatation_generator(top_kernel)

print("-"*20)
print(top_kernel, top_ll)
