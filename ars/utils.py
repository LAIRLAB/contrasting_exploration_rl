# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.
# and
# https://github.com/modestyachts/ARS/blob/master/code/utils.py

import numpy as np
from envs.LQR.LQR import LQREnv
import gym

def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)



def batched_weighted_sum(weights, vecs, batch_size):
    total = 0
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size),
                                         itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float64),
                        np.asarray(batch_vecs, dtype=np.float64))
        num_items_summed += len(batch_weights)
    return total, num_items_summed


def make_env(params, seed):
    if params['env_name'] == 'LQR':
        env = LQREnv(x_dim=params['ob_dim'], u_dim=params['ac_dim'], seed=seed, T=params['rollout_length'], noise_cov=params['noise_cov'])
        return env
    else:
        env = gym.make(params['env_name'])
        env.seed(seed)
        return env
