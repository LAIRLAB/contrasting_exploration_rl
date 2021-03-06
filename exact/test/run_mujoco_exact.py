import numpy as np
from exact.exact import run_exact
import ray
import argparse
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Swimmer-v2')
    parser.add_argument('--n_directions', '-nd', type=int, default=1)
    parser.add_argument('--deltas_used', '-du', type=int, default=1)
    parser.add_argument('--step_size', '-s', type=float, default=0.02)
    parser.add_argument('--delta_std', '-std', type=float, default=1e-2)
    parser.add_argument('--n_workers', '-e', type=int, default=1)
    parser.add_argument('--rollout_length', '-r', type=int, default=10)
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--dir_path', type=str, default='data')
    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='NoFilter')
    parser.add_argument('--one_point', action='store_true')
    parser.add_argument('--coord_descent', action='store_true')
    parser.add_argument('--tuning', action='store_true')
    parser.add_argument('--max_num_steps', type=int, default=1e4)
    parser.add_argument('--non_stationary', action='store_true')
    # horizon parameters
    parser.add_argument('--h_start', type=int, default=1)
    parser.add_argument('--h_end', type=int, default=16)
    parser.add_argument('--h_bin', type=int, default=1)


    args = parser.parse_args()
    params = vars(args)

    params['one_point'] = True
    params['coord_descent'] = True

    ray.init()

    filename = 'data/exact_tuning_'+params['env_name']+'_' + str(args.h_start) + '_' + str(args.h_end) + '_' + str(args.h_bin) +'.pkl'
    _, tuned_params = pickle.load(open(filename, 'rb'))
    ss, nd, ntd, pt = tuned_params

    np.random.seed(params['seed'])
    num_random_seeds = 10
    test_param_seed = list(np.random.randint(low=1, high=1e8, size=num_random_seeds))

    horizons = list(range(args.h_start, args.h_end, args.h_bin))

    result_table = np.zeros((num_random_seeds, len(horizons)))
    for seed_id, seed in enumerate(test_param_seed):
        params['seed'] = seed
        for h_id, h in enumerate(horizons):
            params['step_size'] = ss[h_id]
            params['n_directions'] = nd[h_id]
            params['deltas_used'] = ntd[h_id]
            params['delta_std'] = pt[h_id]
            params['rollout_length'] = h
            params['max_num_steps'] = 1e4*h
            print('Seed: %d, Horizon: %d, Step Size: %f, Num directions: %d, Used directions: %d, Perturbation: %f' % (seed, h, ss[h_id], nd[h_id], ntd[h_id], pt[h_id]))
            result_table[seed_id, h_id] = ray.get(run_exact.remote(params))

    filename = 'data/exact_results_'+ params['env_name']+'_'+ str(args.h_start) + '_' + str(args.h_end) + '_' + str(args.h_bin) +'.pkl'
    pickle.dump(result_table, open(filename, 'wb'))

if __name__ == '__main__':
    main()
