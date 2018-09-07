# Code copied from https://github.com/modestyachts/ARS/blob/master/code/policies.py
'''
Policy class for computing action from weights and observation vector. 
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''


import numpy as np
from ars.filter import get_filter

class Policy(object):

    def __init__(self, policy_params):

        self.ob_dim = policy_params['ob_dim']
        self.ac_dim = policy_params['ac_dim']
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.ob_dim,))
        self.update_filter = True
        
    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params, seed):
        Policy.__init__(self, policy_params)
        self.weights = np.random.RandomState(seed).randn(self.ac_dim, self.ob_dim) * 0.1

    def act(self, ob):
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux
        

class LinearNonStationaryPolicy(Policy):
    """
    Linear non-stationary policy class that computes action as <w_t, ob>
    """

    def __init__(self, policy_params, seed):
        Policy.__init__(self, policy_params)
        assert policy_params['non_stationary'] == True
        self.H = policy_params['H']
        self.weights = np.random.RandomState(seed).randn(self.H, self.ac_dim, self.ob_dim) * 0.1

    def act(self, ob, t):
        assert t < self.H
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights[t, :], ob)

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux
