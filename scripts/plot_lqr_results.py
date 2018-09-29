import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle

parser = argparse.ArgumentParser()

args = parser.parse_args()
params = vars(args)

ars_filename = 'saved_data/ars_results_lqr.pkl'

noise_cov = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]


ars_results = pickle.load(open(ars_filename, 'rb'))

mean_ars = np.mean(ars_results, axis=0)
std_ars = np.std(ars_results, axis=0) / np.sqrt(ars_results.shape[0])

mean_ars = mean_ars / 1e4
std_ars = std_ars / 1e4

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

plt.plot(noise_cov, mean_ars, color='red', label='ARS', linewidth=2)
plt.fill_between(noise_cov, np.maximum(0, mean_ars - std_ars), np.minimum(1e6, mean_ars + std_ars), facecolor='red', alpha=0.2)

plt.xscale('log')
plt.yscale('linear')
plt.xlabel('Standard deviation of the noise in LQR dynamics')
plt.ylabel('Number of samples needed (in multiples of $10^4$)')
plt.title('Plot of number of samples needed vs standard deviation \nof noise in LQR dynamics')

plt.legend()

filename = 'plt_lqr'
plt.savefig('plot/'+filename+'.pdf', format='pdf')
plt.savefig('plot/'+filename+'.png', format='png')

# plt.show()
