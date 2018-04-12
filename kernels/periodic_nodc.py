import gpflow as gpf
import numpy as np
import tensorflow as tf
import scipy.special


class PeriodicNoDC(gpf.kernels.Periodic):
    @gpf.params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D

        r = np.pi * (f - f2) / self.period
        r = tf.reduce_sum(tf.square(tf.sin(r) / self.lengthscales), 2)
        r = tf.exp(-0.5 * r)

        dc_adjustment = np.pi * tf.exp(-tf.square(1 / self.lengthscales)) * \
            tf.py_func(scipy.special.iv, [0, tf.square(1 / self.lengthscales)], tf.float64)
        dc_adjustment = tf.reduce_sum(dc_adjustment)

        # dc_adjustment = 0

        output = self.variance * (r - dc_adjustment) / (1 - dc_adjustment)
        return output
