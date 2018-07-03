#!/usr/bin/env python3
import numpy as np
import os
import gpflow as gpf
import joblib

from kernels import kernels_abstract, kernel_defs, mutate
from report.time_series_report import kernel_description

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def test_kernel(kernel_wrapper, x_vals, y_vals):
    model = gpf.models.GPR(x_vals, y_vals, kern=kernel_wrapper.gpf_kernel)
    model.likelihood.variance = 0.001
    gpf.train.ScipyOptimizer().minimize(model)
    ll = model.likelihood_tensor.eval(session=model.enquire_session())
    return dict(kernel=kernel_wrapper.clone(), loglik=ll)


def center(arr):
    arr -= np.mean(arr)
    arr /= np.std(arr)
    return arr


n_steps = 3

results = []
seen = set()

data = np.load("data/co2.npz")
n_data = 25
x, y = center(data['x'][:n_data]).reshape(-1, 1), center(
    data['y'][:n_data]).reshape(-1, 1)

base_kernels = [kernels_abstract.KernelWrapper(kernel_defs.SEKernel()),
                kernels_abstract.KernelWrapper(kernel_defs.LinKernel()),
                kernels_abstract.KernelWrapper(kernel_defs.PerKernel())]

prospective_kernels = base_kernels

with joblib.Parallel(n_jobs=2) as para:
    for step in range(n_steps):
        print("step {}".format(step))
        to_try = []
        for m in prospective_kernels:
            m.simplify()
            if str(m) not in seen:
                to_try.append(m)

        print("Seen {}".format(seen))
        print("-" * 20)
        print("Kernels to try {}".format(to_try))
        print("-" * 20)
        seen.update((str(m) for m in to_try))

        r = para(joblib.delayed(test_kernel)(m, x, y) for m in to_try)

        results += r
        results = sorted(results, key=lambda z: z['loglik'], reverse=True)

        print("Top kernel is: %s, log likelihood: %f, params: %s" %
              (str(results[0]['kernel']),
               results[0]['loglik'],
               str(np.array([p.read_value() for p in results[0]['kernel'].gpf_kernel.parameters]))))

        print("=" * 20)

        prospective_kernels = mutate.mutation_generator(results[0]['kernel'])

print("Search finished")

report_vars = {'dataset_name': 'co2'}
report_vars.update(kernel_description(results[0]['kernel']))
