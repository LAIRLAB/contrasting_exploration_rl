# Code copied and adapted from https://github.com/modestyachts/ARS/blob/master/code/ars.py
import numpy as np
import ray
import os
import time
import ars.logz as logz
from ars.utils import *
from ars.shared_noise import *
from ars.worker import *
from ars.policies import *
import ars.optimizers as optimizers


class ARSLearner(object):

    def __init__(self, policy_params, logdir, params):
        
        # logz.configure_output_dir(logdir)
        # logz.save_params(params)

        self.env = make_env(params, seed=params['seed'])
        self.is_lqr = params['env_name'] == 'LQR'

        self.timesteps = 0
        self.action_size = self.env.action_space.shape[0]
        self.ob_size = self.env.observation_space.shape[0]
        self.num_deltas = params['n_directions']
        self.deltas_used = params['deltas_used']
        self.rollout_length = params['rollout_length']
        self.step_size = params['step_size']
        self.delta_std = params['delta_std']
        self.logdir = logdir
        self.shift = params['shift']
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')
        self.seed = params['seed']
        self.tuning = params['tuning']

        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed=self.seed+3)

        self.num_workers = params['n_workers']
        self.workers = [Worker.remote(self.seed + 7*i,
                                      policy_params,
                                      deltas_id,
                                      params) for i in range(self.num_workers)]

        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params, seed=params['seed'])
            self.w_policy = self.policy.get_weights()
        else:
            raise NotImplementedError

        self.optimizer = optimizers.SGD(self.w_policy, self.step_size)

    def aggregate_rollouts(self, num_rollouts=None, evaluate=False):

        num_deltas = num_rollouts
        if num_deltas is None:
            num_deltas = self.num_deltas

        policy_id = ray.put(self.w_policy)

        num_rollouts_per_worker = int(num_deltas / self.num_workers)

        rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                                                     num_rollouts=num_rollouts_per_worker,
                                                     shift=self.shift,
                                                     evaluate=evaluate) for worker in self.workers]
        rollout_ids_two = [worker.do_rollouts.remote(policy_id,
                                                     num_rollouts=1,
                                                     shift=self.shift,
                                                     evaluate=evaluate) for worker in self.workers[:(num_deltas % self.num_workers)]]

        results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)

        rollout_rewards, deltas_idx = [], []

        for result in results_one+results_two:
            if not evaluate:
                self.timesteps += result['steps']
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        deltas_idx, rollout_rewards = np.array(deltas_idx), np.array(rollout_rewards, dtype=np.float64)

        if evaluate:
            return rollout_rewards

        max_rewards = np.max(rollout_rewards, axis=1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas

        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100 * (1 - self.deltas_used / self.num_deltas))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx, :]

        rollout_rewards /= np.std(rollout_rewards)

        g_hat, count = batched_weighted_sum(rollout_rewards[:, 0] - rollout_rewards[:, 1],
                                            (self.deltas.get(idx, self.w_policy.size) for idx in deltas_idx),
                                            batch_size=500)

        g_hat /= deltas_idx.size
        return g_hat

    def train_step(self):
        g_hat = self.aggregate_rollouts()
        self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)
        return

    def train(self, max_num_steps):
        start = time.time()
        # for i in range(num_iter):
        i = 0
        while self.timesteps < max_num_steps:            
            self.train_step()

            if ((i+1) % 10 == 0) and not self.tuning:

                rewards = self.aggregate_rollouts(num_rollouts=100, evaluate=True)
                w = ray.get(self.workers[0].get_weights_plus_stats.remote())                
                np.savez(self.logdir + '/lin_policy_plus', w)

                # print(sorted(self.params.items()))
                print
                logz.log_tabular("Time", time.time() - start)
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("AverageReward", np.mean(rewards))
                logz.log_tabular("StdRewards", np.std(rewards))
                logz.log_tabular("MaxRewardRollout", np.max(rewards))
                logz.log_tabular("MinRewardRollout", np.min(rewards))
                logz.log_tabular("timesteps", self.timesteps)
                if self.is_lqr:
                    cost = self.env.evaluate_policy(self.w_policy)
                    logz.log_tabular("optimal cost", self.env.optimal_cost)
                    logz.log_tabular("cost", cost)
                logz.dump_tabular()

            # Check for convergence
            if self.close_to_optimal() and self.is_lqr:
                return self.timesteps

            # get statistics from all workers
            for j in range(self.num_workers):
                self.policy.observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
            self.policy.observation_filter.stats_increment()

            # make sure master filter buffer is clear
            self.policy.observation_filter.clear_buffer()
            # sync all workers
            filter_id = ray.put(self.policy.observation_filter)
            setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
            # waiting for sync of all workers
            ray.get(setting_filters_ids)
         
            increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
            # waiting for increment of all workers
            ray.get(increment_filters_ids)
            i += 1
        return self.timesteps

    def close_to_optimal(self):
        if np.abs(self.env.evaluate_policy(self.w_policy) - self.env.optimal_cost) / self.env.optimal_cost < 0.05:
            return True
        return False

@ray.remote
def run_ars(params):

    dir_path = params['dir_path']

    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    env = make_env(params, seed=params['seed'])
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    policy_params={'type':'linear',
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim}

    ARS = ARSLearner(policy_params=policy_params,
                     logdir=logdir,
                     params=params)                     
        
    num_steps = ARS.train(params['max_num_steps'])
       
    return num_steps
