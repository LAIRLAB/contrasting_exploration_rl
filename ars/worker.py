import ray
import numpy as np
from ars.utils import *
from ars.shared_noise import *
from ars.policies import *


@ray.remote
class Worker(object):
    def __init__(self, seed, policy_params, deltas, params):

        self.env = make_env(params, seed=params['seed'])  # NOTE: All envs should use the same seed for LQR

        self.deltas = SharedNoiseTable(deltas, seed=seed+7)
        self.policy_params = policy_params

        if policy_params['type'] == 'linear' and not policy_params['non_stationary']:
            self.policy = LinearPolicy(policy_params, seed=params['seed'])
        elif policy_params['type'] == 'linear' and policy_params['non_stationary']:
            self.policy = LinearNonStationaryPolicy(policy_params, seed=params['seed'])
        else:
            raise NotImplementedError

        self.delta_std = params['delta_std']
        self.rollout_length = params['rollout_length']
        self.one_point = params['one_point']

    def get_weights_plus_stats(self):
        return self.policy.get_weights_plus_stats()

    def rollout(self, shift=0., rollout_length=None):
        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0
        steps = 0

        ob = self.env.reset()
        for i in range(rollout_length):
            if self.policy_params['non_stationary']:
                action = self.policy.act(ob, i)
            else:                
                action = self.policy.act(ob)
            ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if done:
                break
        return total_reward, steps

    def do_rollouts(self, w_policy, num_rollouts=1, shift=0., evaluate=False):
        rollout_rewards, deltas_idx = [], []
        steps = 0

        for i in range(num_rollouts):
            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)

                self.policy.update_filter = False

                reward, _ = self.rollout(rollout_length=self.rollout_length)
                rollout_rewards.append(reward)
            else:
                idx, delta = self.deltas.get_delta(w_policy.size)

                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)

                self.policy.update_filter = True

                self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps = self.rollout(shift=shift)

                if not self.one_point:                    
                    self.policy.update_weights(w_policy - delta)
                    neg_reward, neg_steps = self.rollout(shift=shift)
                else:
                    neg_reward = 0.
                    neg_steps = 0.

                steps += pos_steps + neg_steps
                rollout_rewards.append([pos_reward, neg_reward])

        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, 'steps': steps}

    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()

    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return
    
