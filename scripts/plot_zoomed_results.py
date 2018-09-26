import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--env_name', type=str, default='Swimmer-v2')
parser.add_argument('--h_start', type=int, default=1)
parser.add_argument('--h_end', type=int, default=202)
parser.add_argument('--h_bin', type=int, default=20)
parser.add_argument('--h_start_z', type=int, default=1)
parser.add_argument('--h_end_z', type=int, default=11)
parser.add_argument('--h_bin_z', type=int, default=1)

args = parser.parse_args()
params = vars(args)

ars_filename = 'data/ars_results_'+params['env_name']+'_'+ str(args.h_start) + '_' + str(args.h_end) + '_' + str(args.h_bin) +'.pkl'
exact_filename = 'data/exact_results_' + params['env_name']+'_'+ str(args.h_start) + '_' + str(args.h_end) + '_' + str(args.h_bin) +'.pkl'

ars_filename_z = 'data/ars_results_'+params['env_name']+'_'+ str(args.h_start_z) + '_' + str(args.h_end_z) + '_' + str(args.h_bin_z) +'.pkl'
exact_filename_z = 'data/exact_results_' + params['env_name']+'_'+ str(args.h_start_z) + '_' + str(args.h_end_z) + '_' + str(args.h_bin_z) +'.pkl'

horizons = list(range(args.h_start, args.h_end, args.h_bin))
horizons_z = list(range(args.h_start_z, args.h_end_z, args.h_bin_z))

ars_results = pickle.load(open(ars_filename, 'rb'))
exact_results = pickle.load(open(exact_filename, 'rb'))

ars_results_z = pickle.load(open(ars_filename_z, 'rb'))
exact_results_z = pickle.load(open(exact_filename_z, 'rb'))

mean_ars = np.mean(ars_results, axis=0)
std_ars = np.std(ars_results, axis=0) / np.sqrt(ars_results.shape[0])

mean_exact = np.mean(exact_results, axis=0)
std_exact = np.std(exact_results, axis=0) / np.sqrt(exact_results.shape[0])

mean_ars_z = np.mean(ars_results_z, axis=0)
std_ars_z = np.std(ars_results_z, axis=0) / np.sqrt(ars_results_z.shape[0])

mean_exact_z = np.mean(exact_results_z, axis=0)
std_exact_z = np.std(exact_results_z, axis=0) / np.sqrt(exact_results_z.shape[0])

fig, ax = plt.subplots()

ax.plot(horizons, mean_ars, color='red', label='ARS', linewidth=2)
ax.fill_between(horizons, mean_ars - std_ars, mean_ars + std_ars, facecolor='red', alpha=0.2)

ax.plot(horizons, mean_exact, color='blue', label='ExAct', linewidth=2)
ax.fill_between(horizons, mean_exact - std_exact, mean_exact + std_exact, facecolor='blue', alpha=0.2)

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
axins = zoomed_inset_axes(ax, 10, loc=2)

axins.plot(horizons_z, mean_ars_z, color='red', label='ARS', linewidth=2)
axins.fill_between(horizons_z, mean_ars_z - std_ars_z, mean_ars_z + std_ars_z, facecolor='red', alpha=0.2)

axins.plot(horizons_z, mean_exact_z, color='blue', label='ExAct', linewidth=2)
axins.fill_between(horizons_z, mean_exact_z - std_exact_z, mean_exact_z + std_exact_z, facecolor='blue', alpha=0.2)

x1, x2, y1, y2 = 1, 10, 0, 10
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

plt.yticks(visible=False)
plt.xticks(visible=False)

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# plt.legend()

plt.show()
