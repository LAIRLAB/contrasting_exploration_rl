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
    parser.add_argument('--n_workers', '-e', type=int, default=2)
    parser.add_argument('--rollout_length', '-r', type=int, default=10)
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--dir_path', type=str, default='data')
    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='NoFilter')
    parser.add_argument('--one_point', action='store_true')
    parser.add_argument('--tuning', action='store_true')
    parser.add_argument('--max_num_steps', type=int, default=1e5)
    # horizon parameters
    parser.add_argument('--h_start', type=int, default=1)
    parser.add_argument('--h_end', type=int, default=21)
    parser.add_argument('--h_bin', type=int, default=2)
    # tuning parameters
    parser.add_argument('--num_random_seeds', type=int, default=5)
    # convergence parameters
    parser.add_argument('--epsilon', type=float, default=1e-3)

    args = parser.parse_args()
    params = vars(args)

    ray.init()

    stepsizes = [5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    num_directions = [1]
    num_top_directions = [1]
    perturbations = [1e-4]
    '''
    stepsizes = [1e-4, 5e-4, 1e-3]
    num_directions = [1, 5, 10]  # NOTE: Always choses 1 direction
    num_top_directions = [1, 5, 10]  # NOTE: Always choses 1 direction
    perturbations = [1e-4, 5e-4, 1e-3]
    '''

    horizons = list(range(args.h_start, args.h_end, args.h_bin))


    initial_seed = 100
    np.random.seed(params['seed'])
    tune_param_seed = list(np.random.randint(low = 1, high = 1e8,size = args.num_random_seeds))
    params['tuning'] = True
    
    result_table = np.zeros((len(tune_param_seed), len(horizons), len(stepsizes), len(num_directions), len(num_top_directions), len(perturbations)))
    result_table = list(result_table.flatten())

    prev_c = 0
    c = 0
    infeasible = False
    for seed in tune_param_seed:
        params['seed'] = seed
        for h in horizons:
            params['rollout_length'] = h
            for s in stepsizes:
                params['step_size'] = s
                for nd in num_directions:
                    params['n_directions'] = nd
                    for ntd in num_top_directions:
                        if ntd > nd:
                            infeasible = True
                        params['deltas_used'] = ntd
                        for p in perturbations:
                            params['delta_std'] = p
                            print('Seed: %d, Horizon: %d, Step Size: %f, Num directions: %d, Used directions: %d, Perturbation: %f' % (seed, h, s, nd, ntd, p))
                            if not infeasible:                                    
                                result_table[c] = run_exact.remote(params)
                            else:
                                result_table[c] = ray.put(float('inf'))
                            c += 1
                        infeasible = False
            result_table[prev_c:c] = ray.get(result_table[prev_c:c])
            print(result_table[prev_c:c])
            prev_c = c


    result_table = np.array(result_table).reshape(len(tune_param_seed), len(horizons), len(stepsizes), len(num_directions), len(num_top_directions), len(perturbations))
    result_table = np.mean(result_table, axis=0)
    min_indices = np.array([np.unravel_index(np.argmin(result_table[i, :]), result_table[i, :].shape) for i in range(len(horizons))])
    ss = np.array(stepsizes)[min_indices[:, 0]]
    nd = np.array(num_directions)[min_indices[:, 1]]
    ntd = np.array(num_top_directions)[min_indices[:, 2]]
    pt = np.array(perturbations)[min_indices[:, 3]]
    filename = 'data/exact_tuning_lqr_' + str(args.h_start) + '_' + str(args.h_end) + '_' + str(args.h_bin) +'.pkl'
    pickle.dump((result_table, (ss, nd, ntd, pt)), open(filename, 'wb'))

if __name__ == '__main__':
    main()
