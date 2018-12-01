import numpy as np
from ars.ars import run_ars
import ray
import argparse
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='LQR')
    parser.add_argument('--ob_dim', type=int, default=100)
    parser.add_argument('--ac_dim', type=int, default=1)
    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--n_directions', '-nd', type=int, default=10)
    parser.add_argument('--deltas_used', '-du', type=int, default=10)
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--delta_std', '-std', type=float, default=.03)
    parser.add_argument('--n_workers', '-e', type=int, default=1)
    parser.add_argument('--rollout_length', '-r', type=int, default=100)
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--dir_path', type=str, default='data')
    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='NoFilter')
    parser.add_argument('--one_point', action='store_true')
    parser.add_argument('--tuning', action='store_true')
    parser.add_argument('--max_num_steps', type=float, default=1e6)
    parser.add_argument('--non_stationary', action='store_true')
    # horizon parameters
    # parser.add_argument('--h_start', type=int, default=1)
    # parser.add_argument('--h_end', type=int, default=302)
    # parser.add_argument('--h_bin', type=int, default=20)
    # Convergence parameters
    parser.add_argument('--epsilon', type=float, default=5e-2)
    parser.add_argument('--noise_cov', type=float, default=0.01)

    args = parser.parse_args()
    params = vars(args)

    ray.init()

    filename = 'data/ars_tuning_lqr.pkl'
    _, tuned_params = pickle.load(open(filename, 'rb'))
    ss, nd, ntd, pt = tuned_params

    np.random.seed(params['seed'])
    num_random_seeds = 10
    test_param_seed = list(np.random.randint(low=1, high=1e8, size=num_random_seeds))

    # horizons = list(range(args.h_start, args.h_end, args.h_bin))
    noise_cov = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

    # ss = [0.01 for _ in range(len(horizons))]
    # nd = ntd = [1 for _ in range(len(horizons))]
    # pt = [1e-4 for _ in range(len(horizons))]

    result_table = np.zeros((num_random_seeds, len(noise_cov)))
    for seed_id, seed in enumerate(test_param_seed):
        params['seed'] = seed
        for nc_id, nc in enumerate(noise_cov):
            params['step_size'] = ss[nc_id]
            params['n_directions'] = nd[nc_id]
            params['deltas_used'] = ntd[nc_id]
            params['delta_std'] = pt[nc_id]
            params['noise_cov'] = nc
            print('Seed: %d, Noise cov: %f, Step Size: %f, Num directions: %d, Used directions: %d, Perturbation: %f' % (seed, nc, ss[nc_id], nd[nc_id], ntd[nc_id], pt[nc_id]))
            result_table[seed_id, nc_id] = ray.get(run_ars.remote(params))

    filename = 'data/ars_results_lqr.pkl'
    pickle.dump(result_table, open(filename, 'wb'))

if __name__ == '__main__':
    main()
