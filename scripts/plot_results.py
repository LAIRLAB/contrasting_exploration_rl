import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--env_name', type=str, default='Swimmer-v2')
parser.add_argument('--h_start', type=int, default=1)
parser.add_argument('--h_end', type=int, default=16)
parser.add_argument('--h_bin', type=int, default=1)
parser.add_argument('--saved', action='store_true')

args = parser.parse_args()
params = vars(args)

if params['saved']:
    directory = 'saved_data'
else:
    directory = 'data'

ars_filename = directory+'/ars_results_'+params['env_name']+'_'+ str(args.h_start) + '_' + str(args.h_end) + '_' + str(args.h_bin) +'.pkl'
exact_filename = directory+'/exact_results_' + params['env_name']+'_'+ str(args.h_start) + '_' + str(args.h_end) + '_' + str(args.h_bin) +'.pkl'

horizons = list(range(args.h_start, args.h_end, args.h_bin))

ars_results = pickle.load(open(ars_filename, 'rb'))
exact_results = pickle.load(open(exact_filename, 'rb'))

mean_ars = np.mean(ars_results, axis=0)
std_ars = np.std(ars_results, axis=0) / np.sqrt(ars_results.shape[0])

mean_exact = np.mean(exact_results, axis=0)
std_exact = np.std(exact_results, axis=0) / np.sqrt(exact_results.shape[0])

plt.plot(horizons, mean_ars, color='red', label='ARS', linewidth=2)
plt.fill_between(horizons, mean_ars - std_ars, mean_ars + std_ars, facecolor='red', alpha=0.2)

plt.plot(horizons, mean_exact, color='blue', label='ExAct', linewidth=2)
plt.fill_between(horizons, mean_exact - std_exact, mean_exact + std_exact, facecolor='blue', alpha=0.2)

plt.legend()

plt.show()
