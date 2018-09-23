import numpy as np
from ars.ars import run_ars
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
    parser.add_argument('--tuning', action='store_true')
    parser.add_argument('--max_num_steps', type=int, default=1e4)
    parser.add_argument('--non_stationary', action='store_true')
    # horizon parameters
    parser.add_argument('--h_start', type=int, default=1)
    parser.add_argument('--h_end', type=int, default=202)
    parser.add_argument('--h_bin', type=int, default=20)
    # tuning parameters
    parser.add_argument('--num_random_seeds', type=int, default=3)

    args = parser.parse_args()
    params = vars(args)

    ray.init()

    stepsizes = [1e-2, 3e-2, 5e-2, 8e-2, 1e-1]
    # stepsizes = [1e-3, 3e-3, 5e-3, 8e-3, 1e-4]
    directions = [5, 10, 20]
    perturbations = [0.03, 0.05, 0.08, 0.1]
    horizons = list(range(args.h_start, args.h_end, args.h_bin))
    # FIX: Using just 1 direction, no normalization and 0.01 perturbation


    initial_seed = 100
    np.random.seed(params['seed'])
    tune_param_seed = list(np.random.randint(low = 1, high = 1e8,size=args.num_random_seeds))
    params['tuning'] = True
    
    result_table = np.zeros((len(tune_param_seed), len(horizons), len(stepsizes), len(directions), len(perturbations)))
    result_table = list(result_table.flatten())

    prev_c = 0
    c = 0
    for seed in tune_param_seed:
        params['seed'] = seed
        for h in horizons:
            params['rollout_length'] = h
            params['max_num_steps'] = 1e4 * h
            for s in stepsizes:
                params['step_size'] = s
                for nd in directions:
                    params['n_directions'] = params['deltas_used'] = nd
                    ntd = nd
                    for p in perturbations:
                        params['delta_std'] = p
                        print('Seed: %d, Horizon: %d, Step Size: %f, Num directions: %d, Used directions: %d, Perturbation: %f' % (seed, h, s, nd, ntd, p))
                        result_table[c] = run_ars.remote(params)
                        c += 1
                
                    result_table[prev_c:c] = ray.get(result_table[prev_c:c])
                    print(result_table[prev_c:c])
                    prev_c = c


    result_table = np.array(result_table).reshape(len(tune_param_seed), len(horizons), len(stepsizes), len(directions), len(perturbations))
    result_table = np.mean(result_table, axis=0)
    max_indices = np.array([np.unravel_index(np.argmax(result_table[i, :]), result_table[i, :].shape) for i in range(len(horizons))])
    ss = np.array(stepsizes)[max_indices[:, 0]]
    nd = np.array(directions)[max_indices[:, 1]]
    ntd = nd
    pt = np.array(perturbations)[max_indices[:, 2]]
    filename = 'data/ars_tuning_'+str(params['env_name'])+'_'+ str(args.h_start) + '_' + str(args.h_end) + '_' + str(args.h_bin) +'.pkl'
    pickle.dump((result_table, (ss, nd, ntd, pt)), open(filename, 'wb'))

if __name__ == '__main__':
    main()
