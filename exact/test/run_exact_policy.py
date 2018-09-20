import argparse
from exact.exact import run_exact
import ray
import socket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Swimmer-v2')
    # parser.add_argument('--ob_dim', type=int, default=100)
    # parser.add_argument('--ac_dim', type=int, default=1)
    parser.add_argument('--n_iter', '-n', type=int, default=1000)
    parser.add_argument('--n_directions', '-nd', type=int, default=10)
    parser.add_argument('--deltas_used', '-du', type=int, default=10)
    parser.add_argument('--step_size', '-s', type=float, default=1e-3)
    parser.add_argument('--delta_std', '-std', type=float, default=0.05)
    parser.add_argument('--n_workers', '-e', type=int, default=1)
    parser.add_argument('--rollout_length', '-r', type=int, default=5)
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--dir_path', type=str, default='data')
    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='NoFilter')
    parser.add_argument('--one_point', action='store_true')
    parser.add_argument('--tuning', action='store_true')
    parser.add_argument('--max_num_steps', type=int, default=1e6)
    
    # ray.init()
    # ray.init(redis_address="192.168.1.115:6379")
    ray.init()
    
    args = parser.parse_args()
    params = vars(args)
    # FIX: one point always true
    params['one_point'] = True
    ray.get(run_exact.remote(params))


if __name__ == '__main__':
    main()
