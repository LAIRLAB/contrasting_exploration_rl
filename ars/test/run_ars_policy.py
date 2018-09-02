import argparse
from ars.ars import run_ars
import ray
import socket


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
    parser.add_argument('--n_workers', '-e', type=int, default=18)
    parser.add_argument('--rollout_length', '-r', type=int, default=10)
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='linear')
    parser.add_argument('--dir_path', type=str, default='data')
    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='NoFilter')

    # local_ip = socket.gethostbyname(socket.gethostname())
    # ray.init(redis_address=local_ip+":6379")
    ray.init()
    
    args = parser.parse_args()
    params = vars(args)
    run_ars(params)


if __name__ == '__main__':
    main()
