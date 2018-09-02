import ray
import numpy as np
from ars.utils import *
from ars.shared_noise import *
from ars.policies import *


@ray.remote
class Worker(object):
    def __init__(self, seed, policy_params, deltas, params):

        self.env = make_env(params, seed)

        self.deltas = SharedNoiseTable(deltas, seed=seed+7)
        self.policy_params = policy_params

        self.ac_dim = self.env.action_space.shape[0]

        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
        else:
            raise NotImplementedError

        self.delta_std = params['delta_std']
        self.rollout_length = params['rollout_length']
        self.seed = seed

    def get_weights_plus_stats(self):
        return self.policy.get_weights_plus_stats()

    def rollout(self, shift=0., rollout_length=None, sampled_t=None, noise=None):
        if rollout_length is None:
            rollout_length = self.rollout_length

        perturbation = True
        if sampled_t is None or noise is None:
            perturbation = False
        elif sampled_t < 0 or sampled_t >= rollout_length:
            raise Exception('Invalid sampled time-step')

        total_reward = 0
        steps = 0
        sampled_obs = None

        ob = self.env.reset()
        for i in range(rollout_length):
            action = self.policy.act(ob)
            if perturbation and i == sampled_t:
                sampled_obs = ob.copy()
                ob, reward, done, _ = self.env.step(action + noise)
            else:
                ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if done:
                break
        return total_reward, steps, sampled_obs

    def do_rollouts(self, w_policy, num_rollouts=1, shift=0., evaluate=False):
        rollout_rewards, deltas_idx, obs, sampled_ts = [], [], [], []
        steps = 0

        for i in range(num_rollouts):
            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)
                sampled_ts.append(-1)
                obs.append(-1)

                self.policy.update_filter = False

                reward, _, _ = self.rollout(rollout_length=self.rollout_length)
                rollout_rewards.append(reward)
            else:
                idx, delta = self.deltas.get_delta(self.ac_dim)

                delta = (self.delta_std * delta)
                deltas_idx.append(idx)

                sampled_t = np.random.uniform(low=0, high=self.rollout_length)

                self.policy.update_filter = True

                # self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps, pos_obs = self.rollout(shift=shift, sampled_t=sampled_t, noise=delta)

                # self.policy.update_weights(w_policy - delta)
                neg_reward, neg_steps, neg_obs = self.rollout(shift=shift, sampled_t=sampled_t, noise=-delta)

                if not np.array_equal(pos_obs, neg_obs):
                    raise NotImplementedError('Only completely deterministic environments are handled')

                steps += pos_steps + neg_steps
                rollout_rewards.append([pos_reward, neg_reward])
                obs.append(pos_obs)
                sampled_ts.append(sampled_t)

        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, 'steps': steps, 'obs': obs, 'sampled_ts': sampled_ts}

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
