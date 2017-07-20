#paper_sims.py
#Trying to recreate simulation results from NIPS 2016 paper
#"Multi-fidelity Multi-armed bandit"

import matplotlib.pyplot as plt
import numpy as np
from mf_agent import Agent
from mf_policy import MF_UCBPolicy, UCBPolicy
from mf_environment import Environment
from mf_bandits import MF_GaussianBandit
from tqdm import tqdm

def make_psi_inv(pow):
	def psi_inv(exploration):
		return np.power(exploration, pow)
	return psi_inv    

if __name__ == '__main__':
	experiments = 13

	#Example 1 - Linear means, Gaussian rewards
	zeta = [0.2, 0.1, 0] 	#bound on fidelity mean over/undershoot
	costs = [1, 10, 100] 	#increasing costs of increasing fidelities
	no_arms = 10  		  	#number of arms in multi-armed bandit
	no_fids = 3				#number of fidelities per arm
	means = np.zeros((no_arms, no_fids))	#matrix of arm,fid means
	#highest fidelity means uniform grid in (0,1)
	high_fid_means = np.random.normal(means[:,0],1.0)
	means[:,no_fids-1] += high_fid_means
	#lower fidelity means sampled uniformly within a +/- zeta band around high fid mean for corresponding arm
	for m in range(no_fids-1):
		#create zeta interval about high fid. mean
		upper_bd = high_fid_means+zeta[m]
		lower_bd = high_fid_means-zeta[m]
		#sample low fidelity means uniformly from the zeta interval
		means[:,m] = np.random.uniform(lower_bd, upper_bd, size=no_arms)	
	#MODIFY the lower fidelity means of the optimal arm
	#to be lower than the corresponding mean of a suboptimal arm
	for m in range(no_fids-1):
		while means[-1,m]>=means[-2,m]:
			means[-1,m] = np.random.uniform(lower_bd[-1], upper_bd[-1], size=1)	
	#plot means
	plt.plot(means, '.')
	plt.show()
	#broadcast zeta into similarly shaped matrix as the means
	zeta = np.broadcast_to(zeta, (no_arms,no_fids))

	#create MA bandit w/ gaussian rewards + these means w/ sigma=0.2
	bandit = MF_GaussianBandit(k=no_arms, m=no_fids, mu=means, sigma=1, zeta=zeta, costs=costs)
	#bandit.plot_rewards()

	#create MF agent implementing UCB alg w/ psi(x)=x^2, rho=2
	psi_inv = make_psi_inv(1/2)
	agent = Agent(bandit, MF_UCBPolicy(rho=2, psi_inv=psi_inv))

	#single fidelity agent on same bandit (highest fidelity)
	agent2 = Agent(bandit, UCBPolicy(rho=2, psi_inv=psi_inv))

	# #Run simulation for MF bandit
	env_mf = Environment(bandit, agent, 'simple MF ex')
	env_single = Environment(bandit, agent2, 'single fid ex')

	cost_constraints = 1000000

	plays, regrets, optimal_pulls = env_mf.run(cost_constraints, experiments)
	plays2, regrets2, optimal_pulls2 = env_single.run(cost_constraints, experiments)
	#print(optimal_pulls)
	#print(optimal_pulls2)
	fig, ax = plt.subplots()
	index = np.arange(zeta.shape[0])
	bar_width=0.1
	arm_plays0 = plt.bar(index, plays[:,0], bar_width,
			 color='b', label='Fid 0 (MF-UCB)')
	arm_plays1 = plt.bar(index + bar_width, plays[:,1], bar_width, 
					color='g', label='Fid 1 (MF-UCB)')
	arm_plays2 = plt.bar(index + 2*bar_width, plays[:,2], bar_width,
					color='r', label='Fid 2 (MF-UCB)')
	arm_plays_sf = plt.bar(index+3*bar_width, plays2[:,2], bar_width, color='k', label='Fid 2 (UCB)')
	plt.xlabel('Arm')
	plt.ylabel('Plays')
	plt.title('Arm play count w/ resources %d' %cost_constraints)
	plt.legend()
	plt.show()

	# #regret vs cost increases 
	# cost_constraints = [0.25*(10**5), 0.5*(10**5), 1*(10**5), 1.5*(10**5), 2*(10**5), 2.5*(10**5),
	# 					 3*(10**5), 3.5*(10**5), 4*(10**5), 4.5*(10**5), 5*(10**5)]
	# regrets = np.zeros_like(cost_constraints)
	# regrets2 = np.`zeros_like(regrets)

	# for k in tqdm(range(len(cost_constraints))):
	# 	plays, regrets[k], optimal_pulls = env_mf.run(cost_constraints[k], experiments)
	# 	plays2, regrets2[k], optimal_pulls2 = env_single.run(cost_constraints[k], experiments)
	# 	#print("optimal pulls w/ MF = %d" %optimal_pulls)
	# 	#print("opt pulls w/ single fid = %d" %optimal_pulls2)

	# #plot regret vs cost
	# plt.plot(cost_constraints, regrets, color='k', marker='s',label='MF-UCB')
	# plt.plot(cost_constraints, regrets2, color='b', marker='o',label='UCB')
	# plt.title('Regret vs Total Cost Constraint')
	# plt.legend()
	# axes = plt.gca()
	# axes.set_xlim([np.amin(cost_constraints), np.amax(cost_constraints)])
	# plt.show()	
