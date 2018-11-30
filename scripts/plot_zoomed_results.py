import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle

# EDIT: Increase font size according to reviwer's comments
plt.rcParams.update({'font.size': 25})

parser = argparse.ArgumentParser()

parser.add_argument('--env_name', type=str, default='Swimmer-v2')
parser.add_argument('--h_start', type=int, default=1)
parser.add_argument('--h_end', type=int, default=16)
parser.add_argument('--h_bin', type=int, default=1)
parser.add_argument('--h_start_z', type=int, default=1)
parser.add_argument('--h_end_z', type=int, default=6)
parser.add_argument('--h_bin_z', type=int, default=1)

args = parser.parse_args()
params = vars(args)

ars_filename = 'saved_data/ars_results_'+params['env_name']+'_'+ str(args.h_start) + '_' + str(args.h_end) + '_' + str(args.h_bin) +'.pkl'
exact_filename = 'saved_data/exact_coord_results_' + params['env_name']+'_'+ str(args.h_start) + '_' + str(args.h_end) + '_' + str(args.h_bin) +'.pkl'

horizons = list(range(args.h_start, args.h_end, args.h_bin))
horizons_z = list(range(args.h_start_z, args.h_end_z, args.h_bin_z))

num_z = len(horizons_z)

ars_results = pickle.load(open(ars_filename, 'rb'))
exact_results = pickle.load(open(exact_filename, 'rb'))

mean_ars = np.mean(ars_results, axis=0)
std_ars = np.std(ars_results, axis=0) / np.sqrt(ars_results.shape[0])

mean_exact = np.mean(exact_results, axis=0)
std_exact = np.std(exact_results, axis=0) / np.sqrt(exact_results.shape[0])

fig, ax = plt.subplots()

ax.plot(horizons, mean_ars, color='red', label='ARS', linewidth=2)
ax.fill_between(horizons, mean_ars - std_ars, mean_ars + std_ars, facecolor='red', alpha=0.2)

ax.plot(horizons, mean_exact, color='blue', label='ExAct', linewidth=2)
ax.fill_between(horizons, mean_exact - std_exact, mean_exact + std_exact, facecolor='blue', alpha=0.2)

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
zoom = None
if args.env_name == 'Swimmer-v2':
    zoom = True
elif args.env_name == 'HalfCheetah-v2':    
    zoom = True # False
else:
    raise NotImplementedError
if zoom:
    
    axins = zoomed_inset_axes(ax, 1.5, loc=2)
    
    axins.plot(horizons_z, mean_ars[:num_z], color='red', label='ARS', linewidth=2)
    axins.fill_between(horizons_z, mean_ars[:num_z] - std_ars[:num_z], mean_ars[:num_z] + std_ars[:num_z], facecolor='red', alpha=0.2)

    axins.plot(horizons_z, mean_exact[:num_z], color='blue', label='ExAct', linewidth=2)
    axins.fill_between(horizons_z, mean_exact[:num_z] - std_exact[:num_z], mean_exact[:num_z] + std_exact[:num_z], facecolor='blue', alpha=0.2)

#x1, x2, y1, y2 = 1, 5, 0, 5
#axins.set_xlim(x1, x2)
#axins.set_ylim(y1, y2)


if zoom:
    axins.set_ylim(0, 2)
    plt.yticks(visible=False)
    plt.xticks(visible=False)

if zoom:    
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax, axins, loc1=4, loc2=3, fc="none", ec="0.5")

ax.set_xlabel('Horizon Length H')
ax.set_ylabel('Mean Return')

ax.legend(loc='lower right')

title = 'Plot of Mean Return vs Horizon Length H for '+str(params['env_name'])
ax.set_title(title)

plt.gcf().set_size_inches([11.16, 8.26])

filename = 'plt_'+str(params['env_name'])+'.pdf'
plt.savefig('plot/'+filename, format='pdf')

filename = 'plt_'+str(params['env_name'])+'.png'
plt.savefig('plot/'+filename, format='png')

#plt.show()
