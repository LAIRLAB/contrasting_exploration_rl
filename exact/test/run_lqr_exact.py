import numpy as np
from exact.exact import run_exact
import ray
import argparse
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='LQR')
    parser.add_argument('--ob_dim', type=int, default=100)
    parser.add_argument('--ac_dim', type=int, default=1)
    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--n_directions', '-nd', type=int, default=8)
    parser.add_argument('--deltas_used', '-du', type=int, default=8)
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--delta_std', '-std', type=float, default=.03)
    parser.add_argument('--n_workers', '-e', type=int, default=1)
    parser.add_argument('--rollout_length', '-r', type=int, default=10)
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--dir_path', type=str, default='data')
    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='NoFilter')
    parser.add_argument('--one_point', action='store_true')
    parser.add_argument('--tuning', action='store_true')
    parser.add_argument('--max_num_steps', type=float, default=1e5)
    # horizon parameters
    parser.add_argument('--h_start', type=int, default=1)
    parser.add_argument('--h_end', type=int, default=202)
    parser.add_argument('--h_bin', type=int, default=20)
    # Convergence parameters
    parser.add_argument('--epsilon', type=float, default=2e-2)

    args = parser.parse_args()
    params = vars(args)

    ray.init()
    # ray.init(redis_address="192.168.1.115:6379")

    filename = 'data/exact_tuning_lqr_' + str(args.h_start) + '_' + str(args.h_end) + '_' + str(args.h_bin) +'.pkl'
    _, tuned_params = pickle.load(open(filename, 'rb'))
    ss, nd, ntd, pt = tuned_params

    horizons = list(range(args.h_start, args.h_end, args.h_bin))

    np.random.seed(params['seed'])
    num_random_seeds = 10
    test_param_seed = list(np.random.randint(low=1, high=1e8, size=num_random_seeds))

    result_table = np.zeros((num_random_seeds, len(horizons)))
    for seed_id, seed in enumerate(test_param_seed):
        params['seed'] = seed
        for h_id, h in enumerate(horizons):
            params['step_size'] = ss[h_id]
            params['n_directions'] = nd[h_id]
            params['deltas_used'] = ntd[h_id]
            params['delta_std'] = pt[h_id]
            params['rollout_length'] = h
            print('Seed: %d, Horizon: %d, Step Size: %f, Num directions: %d, Used directions: %d, Perturbation: %f' % (seed, h, ss[h_id], nd[h_id], ntd[h_id], pt[h_id]))
            result_table[seed_id, h_id] = ray.get(run_exact.remote(params))

    filename = 'data/exact_results_lqr_'+ str(args.h_start) + '_' + str(args.h_end) + '_' + str(args.h_bin) +'.pkl'
    pickle.dump(result_table, open(filename, 'wb'))

if __name__ == '__main__':
    main()
