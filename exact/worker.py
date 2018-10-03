import ray
import numpy as np
from exact.utils import *
from ars.shared_noise import *
from ars.policies import *
from gym.utils import seeding


@ray.remote
class Worker(object):
    def __init__(self, seed, policy_params, deltas, params):

        self.env = make_env(params, seed=params['seed'])  # NOTE: Need to use the same env seed across all workers for LQR

        self.deltas = SharedNoiseTable(deltas, seed=seed+7)
        self.policy_params = policy_params

        self.ac_dim = self.env.action_space.shape[0]

        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params, seed=params['seed'])
        else:
            raise NotImplementedError

        self.delta_std = params['delta_std']
        self.rollout_length = params['rollout_length']
        self.one_point = params['one_point']
        self.coord_descent = params['coord_descent']
        self.seed = seed
        self.params = params
        self.np_random, _ = seeding.np_random(seed)

    def get_weights_plus_stats(self):
        return self.policy.get_weights_plus_stats()

    def rollout(self, shift=0., rollout_length=None, noise=None, sampled_t=None):
        if rollout_length is None:
            rollout_length = self.rollout_length

        perturbation = True
        if noise is None:
            perturbation = False

        total_reward = 0
        steps = 0
        obs = []

        ob = self.env.reset()
        for i in range(rollout_length):
            action = self.policy.act(ob)
            if perturbation and not self.coord_descent:
                obs.append(ob)
                noise_t = noise[i, :]
                ob, reward, done, _ = self.env.step(action + noise_t)
            elif perturbation and self.coord_descent:
                if i == sampled_t:
                    obs.append(ob)
                    ob, reward, done, _ = self.env.step(action + noise)
                else:
                    ob, reward, done, _ = self.env.step(action)
            else:
                ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if done:
                break
        return total_reward, steps, obs

    def do_rollouts(self, w_policy, num_rollouts=1, shift=0., evaluate=False):
        rollout_rewards, deltas_idx, obs = [], [], []
        steps = 0

        for i in range(num_rollouts):
            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)
                obs.append(-1)

                self.policy.update_filter = False

                reward, _, _ = self.rollout(rollout_length=self.rollout_length)
                rollout_rewards.append(reward)
            else:
                self.policy.update_weights(w_policy)

                sampled_t = None
                if not self.coord_descent:                    
                    idx, delta = self.deltas.get_delta(self.ac_dim * self.rollout_length)
                    delta = (self.delta_std * delta).reshape(self.rollout_length, self.ac_dim)
                else:
                    idx, delta = self.deltas.get_delta(self.ac_dim)
                    delta = self.delta_std * delta
                    sampled_t = self.np_random.randint(low=0, high=self.rollout_length)

                deltas_idx.append(idx)

                self.policy.update_filter = True
                
                # self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps, pos_obs = self.rollout(shift=shift, noise=delta, sampled_t=sampled_t)

                # self.policy.update_weights(w_policy - delta)
                if not self.one_point:                    
                    neg_reward, neg_steps, neg_obs = self.rollout(shift=shift, noise=-delta, sampled_t=sampled_t)
                else:
                    neg_reward = 0.
                    neg_steps = 0.
                    neg_obs = pos_obs.copy()

                if not np.array_equal(pos_obs, neg_obs):
                    raise NotImplementedError('Only completely deterministic environments are handled. Use one point for stochastic environments')

                if pos_obs is None or neg_obs is None:
                    raise Exception('Observation not assigned')

                if len(pos_obs) == 0 or len(neg_obs) == 0:
                    raise Exception('Observation not assigned')

                steps += pos_steps + neg_steps
                rollout_rewards.append([pos_reward, neg_reward])
                obs.append(pos_obs)

        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, 'steps': steps, 'obs': obs}

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
