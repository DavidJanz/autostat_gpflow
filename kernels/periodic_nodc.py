import gpflow as gpf
import numpy as np
import tensorflow as tf
import scipy.special


class PeriodicNoDC(gpf.kernels.Periodic):
    @gpf.params_as_tensors
    def K(self, X, X2=None, presliced=False):
        # X = tf.Print(input_=X, data=[X, tf.shape(X)], message="Input (periodicnodc):")
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D

        # self.lengthscales = tf.Print(input_=self.lengthscales,
        #                              data=[self.lengthscales, tf.shape(self.lengthscales)],
        #                              message="Lengthscales (periodicnodc):")

        r = 2 * np.pi * (f - f2) / self.period
        r = tf.case([(tf.greater(self.lengthscales, 1e4), lambda: large_l(r)),
                     (tf.less(self.lengthscales, 3.75), lambda: small_l(r, self.lengthscales))
                     ],
                    default=lambda: medium_l(r, self.lengthscales))

        output = self.variance * r
        # output = tf.Print(input_=output, data=[output, tf.shape(output)], message="Output (periodicnodc): ")
        return output

# %%% https://github.com/jamesrobertlloyd/gpss-research/blob/master/source/matlab/custom-cov/covPeriodicNoDC.m
# K = sqrt(sq_dist(x',z'));
# K = 2*pi*K/p;  # p is the period
# K = sf2*covD(K,ell);  # ell is lengthscale
# return K
#
# function K = covD(D,ell)                   % evaluate covariances from distances
#   if ell>1e4                                              % limit for ell->infty
#     K = cos(D);
#   elseif 1/ell^2<3.75
#     K = exp(cos(D)/ell^2);
#     b0 = besseli(0,1/ell^2);
#     K = (K-b0)/(exp(1/ell^2)-b0);
#   else
#     K = exp((cos(D)-1)/ell^2);
#     b0 = embi0(1/ell^2);
#     K = (K-b0)/(1-b0);
#   end


def large_l(r):
    r = tf.reduce_sum(tf.cos(r), 2)  # not entirely sure about reduce_sum here
    return r


def small_l(r, lengthscales):
    r = tf.reduce_sum(tf.cos(r) / tf.square(lengthscales), 2)
    r = tf.exp(r)
    b0 = tf.py_func(scipy.special.iv, [0, tf.square(1 / lengthscales)], tf.float64)
    r = (r - b0) / (tf.exp(tf.square(1 / lengthscales)) - b0)
    return r


def medium_l(r, lengthscales):
    r = tf.reduce_sum((tf.cos(r) - 1) / tf.square(lengthscales), 2)
    r = tf.exp(r)
    b0 = tf.py_func(embi0, [1 / tf.square(lengthscales)], tf.float64)
    r = (r - b0) / (1 - b0)
    return r


def embi0(x):  # = exp(-x)*besseli(0,x) => 9.8.2 Abramowitz & Stegun
    y = 3.75 / x
    y = (0.39894228 + 0.01328592 * y + 0.00225319 * y ^ 2 - 0.00157565 * y ^ 3
         + 0.00916281 * y ^ 4 - 0.02057706 * y ^ 5 + 0.02635537 * y ^ 6 - 0.01647633 * y ^ 7
         + 0.00392377 * y ^ 8)
    y = y / x ^ 0.5
    return y
