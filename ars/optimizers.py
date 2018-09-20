# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.
# and
# https://github.com/modestyachts/ARS/blob/master/code/optimizers.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# OPTIMIZERS FOR MINIMIZING OBJECTIVES
class Optimizer(object):
    def __init__(self, w_policy):
        self.w_policy = w_policy.flatten()
        self.dim = w_policy.size
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        ratio = np.linalg.norm(step) / (np.linalg.norm(self.w_policy) + 1e-5)
        return self.w_policy + step, ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, pi, stepsize):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize

    def _compute_step(self, globalg):
        step = -self.stepsize * globalg
        return step


class Adam(Optimizer):
    def __init__(self, pi, stepsize):
        Optimizer.__init__(self, pi)
        self.alpha = stepsize
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.gamma = 1 - 1e-8

        self.m_t = np.zeros(self.dim)
        self.v_t = np.zeros(self.dim)
        self.t = 0

    def _compute_step(self, globalg):
        self.t += 1
        self.beta_1_t = self.beta_1 * (self.gamma ** (self.t - 1))
        self.m_t = self.beta_1_t * self.m_t + (1 - self.beta_1_t) * globalg
        self.v_t = self.beta_2 * self.v_t + (1 - self.beta_2) * (globalg**2)

        m_cap = self.m_t / (1 - (self.beta_1**self.t))
        v_cap = self.v_t / (1 - (self.beta_2**self.t))
        
        step = - (self.alpha * m_cap) / (np.sqrt(v_cap) + self.epsilon)
        return step
