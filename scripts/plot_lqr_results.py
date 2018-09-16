import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--h_start', type=int, default=1)
parser.add_argument('--h_end', type=int, default=21)
parser.add_argument('--h_bin', type=int, default=2)

args = parser.parse_args()
params = vars(args)

ars_filename = 'data/ars_results_lqr_'+ str(args.h_start) + '_' + str(args.h_end) + '_' + str(args.h_bin) +'.pkl'
exact_filename = 'data/exact_results_lqr_'+ str(args.h_start) + '_' + str(args.h_end) + '_' + str(args.h_bin) +'.pkl'

horizons = list(range(args.h_start, args.h_end, args.h_bin))

ars_results = pickle.load(open(ars_filename, 'rb'))
exact_results = pickle.load(open(exact_filename, 'rb'))

mean_ars = np.mean(ars_results, axis=0)
std_ars = np.std(ars_results, axis=0) / np.sqrt(ars_results.shape[0])

mean_exact = np.mean(exact_results, axis=0)
std_exact = np.std(exact_results, axis=0) / np.sqrt(exact_results.shape[0])

plt.plot(horizons, mean_ars, color='red', label='ARS', linewidth=2)
plt.fill_between(horizons, np.maximum(0, mean_ars - std_ars), np.minimum(1e8, mean_ars + std_ars), facecolor='red', alpha=0.2)

plt.plot(horizons, mean_exact, color='blue', label='ExAct', linewidth=2)
plt.fill_between(horizons, np.maximum(0, mean_exact - std_exact), np.minimum(1e8, mean_exact + std_exact), facecolor='blue', alpha=0.2)

plt.legend()

plt.show()
