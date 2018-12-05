import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle

# EDIT: Increase font size according to reviwer's comments
plt.rcParams.update({'font.size': 25})

parser = argparse.ArgumentParser()
parser.add_argument('--saved', action='store_true')

args = parser.parse_args()
params = vars(args)

if params['saved']:
    directory = 'saved_data'
else:
    directory = 'data'

ars_filename = directory+'/ars_results_lqr.pkl'
exact_filename =  directory+'/exact_results_lqr.pkl'

noise_cov = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]


ars_results = pickle.load(open(ars_filename, 'rb'))
exact_results = pickle.load(open(exact_filename, 'rb'))

mean_ars = np.mean(ars_results, axis=0)
std_ars = np.std(ars_results, axis=0) / np.sqrt(ars_results.shape[0])

mean_exact = np.mean(exact_results, axis=0)
std_exact = np.std(exact_results, axis=0) / np.sqrt(exact_results.shape[0])

mean_ars = mean_ars / 1e4
std_ars = std_ars / 1e4

mean_exact = mean_exact / 1e4
std_exact = std_exact / 1e4


#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

plt.plot(noise_cov, mean_ars, color='red', label='ARS', linewidth=2)
plt.fill_between(noise_cov, np.maximum(0, mean_ars - std_ars), np.minimum(1e6, mean_ars + std_ars), facecolor='red', alpha=0.2)

plt.plot(noise_cov, mean_exact, color='blue', label='ExAct', linewidth=2)
plt.fill_between(noise_cov, np.maximum(0, mean_exact - std_exact), np.minimum(1e6, mean_exact + std_exact), facecolor='blue', alpha=0.2)

plt.xscale('log')
plt.yscale('linear')
plt.xlabel('Standard deviation of the noise in LQR dynamics')
plt.ylabel('Number of samples (multiples of $10^4$)')
plt.title('Plot of number of samples vs std dev \nof noise in LQR dynamics')

plt.legend()

plt.gcf().set_size_inches([11.16, 8.26])

filename = 'plt_lqr'
plt.savefig('plot/'+filename+'.pdf', format='pdf')
plt.savefig('plot/'+filename+'.png', format='png')

# plt.show()
