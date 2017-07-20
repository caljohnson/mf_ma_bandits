#mf_bandits.py
#multi-fidelity multi-armed bandit classes
#based on
#Byron Galbraith's multi-armed bandit simulator classes
#source: https://github.com/bgalbraith/bandits/blob/master/bandits/bandit.py

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

class MF_MultiArmedBandit(object):
	"""
	A Multi-fidelity Multi-armed Bandit
	with k arms and m fidelities (per each arm)
	and associated costs
	"""
	def __init__(self, k, m, costs):
		self.k = k #number of bandit arms
		self.m = m #number of fidelities per arm
		self.costs = costs #costs(k,m) cost of playing arm k at fidelity m
		self.action_values = np.zeros((k,m)) #expected rewards of each arm
		self.optimal = 0 #optimal arm

	def reset(self):
		self.action_values = np.zeros((self.k, self.m))
		self.optimal = 0

	def pull(self, action):
		return 0


class MF_GaussianBandit(MF_MultiArmedBandit):
	"""
	Gaussian bandits model the reward of a given arm as normal distribution with
	provided mean and standard deviation.
	Each arm k has m fidelities
	Passing in a matrix of mu(k,m) values allows for each arm k and fidelity level m to use
	the corresponding mean
	zeta(k, m) is the bound on the interval about which fidelity m over/undershoots 
	the mean of the highest fidelity - identical across arms k
	"""
	def __init__(self, k, m, mu=0, sigma=1, zeta=0, costs=0):
		super(MF_GaussianBandit, self).__init__(k,m, costs)
		self.mu = mu #mu(k,m) is the mean gaussian reward from arm k at fidelity m
		self.sigma = sigma #sigma(k,m) is the std dev of gaussian rewards
		self.zeta = zeta #zeta(k,m) bound on mean over/undershoot by fidelity m
		self.reset() #resets the bandits to random rewards

	def reset(self):
		#self.action_values = np.random.normal(self.mu, self.sigma, (self.k, self.m))
		self.action_values = self.mu

		#oracle only looks at highest fidelity (index m-1) to select the best arm
		self.optimal = [np.argmax(self.action_values[:,self.m-1]), self.m-1]

	def pull(self, action):
		#pulls the bandit arm action[0] at fidelity action[1]
		return np.random.normal(self.action_values[action[0], action[1]], self.sigma), action==self.optimal
	
	def plot_rewards(self):
		fig, ax = plt.subplots()
		index = np.arange(self.zeta.shape[0])
		bar_width=0.1
		opacity=0.4
		error_config={'ecolor':'0.001'}
		colors = ['b', 'g', 'r', 'k', 'p']
		rews0 = plt.bar(index, self.action_values[:,0], bar_width, alpha=opacity,
			 color='b', yerr=self.zeta[:,0], error_kw=error_config, label='Fid 0')
		rews1 = plt.bar(index + bar_width, self.action_values[:,1], bar_width, alpha=opacity,
					color='g', yerr=self.zeta[:,1], error_kw=error_config, label='Fid 1')
		rews2 = plt.bar(index + 2*bar_width, self.action_values[:,2], bar_width, alpha=opacity,
					color='r', yerr=self.zeta[:,2], error_kw=error_config, label='Fid 2')
		if self.m>=4:
			rews3 = plt.bar(index, self.action_values[:,fid], bar_width, alpha=opacity,
					color='k', yerr=self.zeta[:,3], error_kw=error_config, label='Fid 3')
		if self.m>=5:
			rews4 = plt.bar(index, self.action_values[:,0], bar_width, alpha=opacity,
			 color='p', yerr=self.zeta[:,4], error_kw=error_config, label='Fid 4')
		plt.xlabel('Arm')
		plt.ylabel('Rewards')
		plt.title('Arm rewards')
		plt.legend();plt.show()                