#mytry.py
#My own try at implementing multi-fidelity multi-armed bandits w/ UCB alg
#based on
#Byron Galbraith's example of bernoulli and binomial bandits
#source: https://github.com/bgalbraith/bandits/blob/master/examples/bayesian.py


import matplotlib.pyplot as plt
import numpy as np
from mf_agent import Agent
from mf_policy import MF_UCBPolicy, UCBPolicy
from mf_environment import Environment
from mf_bandits import MF_GaussianBandit
from tqdm import tqdm

class GaussianEx(object):
    """
    Multifidelity multiarmed Bandit
    w/ Gaussian rewards
    High fidelities dist. w/ means mu_k uniform in grid (0,1) or normally dist.
    Fidelity "m" means dist. uniformly within a +- zeta(m) band about mu_k
    """
    def __init__(self, no_arms, no_fids, zeta, costs, high_fid_mean_dist='unif'):
        self.label = 'Multi-armed Bandits - Gaussian ('+high_fid_mean_dist+')'
        self.no_arms = no_arms
        self.no_fids = no_fids
        self.bandit = self.make_bandit(zeta, costs, high_fid_mean_dist)
        self.psi_inv = self.make_psi_inv()
        self.agent = Agent(self.bandit, MF_UCBPolicy(rho=2, psi_inv=self.psi_inv))

    def make_psi_inv(self):
        def psi_inv(exploration):
            return np.power(exploration, 1/2)
        return psi_inv        

    def make_bandit(self,zeta, costs, high_fid_mean_dist):        
        #pick high fidelity means either as a uniform grid in (0,1)
        #or sampled from a N(0,1) dist, sorted low->high
        if high_fid_mean_dist=='unif':
            high_fid_means = np.linspace(0.0,1.0, num=self.no_arms)
            sigma=0.2
        if high_fid_mean_dist=='normal':
            high_fid_means = np.random.normal(0.0, 1.0, size=self.no_arms)
            high_fid_means = np.sort(high_fid_means) 
            sigma=1

        #initialize matrix of all fidelity means
        self.fid_means = np.zeros((self.no_arms,self.no_fids-1))
        for m in range(self.no_fids-1):
            #create zeta interval about high fid. mean
            upper_bd = high_fid_means+zeta[m]
            lower_bd = high_fid_means-zeta[m]
            #sample low fidelity means uniformly from the zeta interval
            self.fid_means[:,m] = np.random.uniform(lower_bd, upper_bd, size=self.no_arms)
        #add high fidelity means to fidelity mean matrix    
        self.fid_means = np.c_[self.fid_means, high_fid_means]
        #save zeta interval into matrix w/ corresponding fidelity, arm positions
        self.zeta = np.broadcast_to(zeta, (self.no_arms, self.no_fids))
        #create MF-MA bandit w/ Gaussian rewards w/ means according to fid. means matrix
        return MF_GaussianBandit(k=self.no_arms, m=self.no_fids, mu=self.fid_means, sigma=sigma, zeta=self.zeta, costs=costs)
    
    def plot_means(self):
        plt.plot(self.fid_means, '.')
        plt.show()
        plt.close()


if __name__ == '__main__':
    experiments = 1000

    #Example 1
    zeta1 = [0.2, 0.1, 0] #bound on fidelity mean over/undershoot
    costs1 = [1, 10, 100] #increasing costs of increasing fidelities
    #create MF-MA bandit w/ gaussian rewards, linear means
    example1 = GaussianEx(no_arms=500, no_fids=3, zeta=zeta1, costs=costs1)
    #show means of all arms and fidelities
    #example1.plot_means()

    #create single-fidelity model for comparison
    example1_single = GaussianEx(no_arms=500, no_fids=1, zeta=[0], costs=[100])

    #Example 2
    zeta2 = [1, 0.5, 0.2, 0] #bound on fidelity mean over/undershoot
    costs2 = [1,5,20,50]    #increasings costs of increasing fidelities
    #create MF-MA bandit w/ gaussian rewards, normally dist. means
    example2 = GaussianEx(no_arms=5, no_fids=4, zeta=zeta2, costs=costs2, high_fid_mean_dist='normal')
    #show means of all arms and fidelities
    example2.plot_means()
    example2_single = GaussianEx(no_arms=5, no_fids=1, zeta=[0], costs=[50], high_fid_mean_dist='normal')
    
    # #Run simulation for Example 1 bandit
    env1 = Environment(example1.bandit, example1.agent, example1.label)
    env1_single = Environment(example1_single.bandit, example1_single.agent, example1_single.label)

    #regret vs cost increases 
    cost_constraints = np.linspace(0.5*(10**5), 5*(10**5), num=10)
    regrets = np.zeros_like(cost_constraints)
    regrets2= np.zeros_like(cost_constraints)

    for k in tqdm(range(len(cost_constraints))):
        plays, regrets[k] = env1.run(cost_constraints[k], experiments)
        plays2, regrets2[k] = env1_single.run(cost_constraints[k], experiments)
        #plot arm+fidelity plays
        #env1.plot_plays(plays)

    #plot regret vs cost
    env1.plot_cost_vs_regret(cost_constraints, regrets)
    #plot regret vs cost
    plt.plot(cost_constraints, regrets, color='b')
    plt.plot(cost_constraints, regrets2, color='r')
    axes = plt.gca()
    axes.set_xlim([np.amin(cost_constraints), np.amax(cost_constraints)])
    axes.set_ylim([np.amin(regrets),np.amax(regrets2)])
    plt.show()

    #Run simulation for Example 2 bandit
    env2 = Environment(example2.bandit, example2.agent, example2.label)
    env2_single = Environment(example2_single.bandit, example2_single.agent, example2_single.label)

    #regret vs cost increases 
    cost_constraints = np.linspace(0.5*(10**5), 5*(10**5), num=10)
    regrets = np.zeros_like(cost_constraints)
    regrets2= np.zeros_like(cost_constraints)

    for k in tqdm(range(len(cost_constraints))):
        plays, regrets[k] = env2.run(cost_constraints[k], experiments)
        plays, regrets2[k] = env2_single.run(cost_constraints[k], experiments)
        #plot arm+fidelity plays
        #env2.plot_plays(plays)

   # env2.plot_cost_vs_regret(cost_constraints, regrets)
    #plot regret vs cost
    plt.plot(cost_constraints, regrets, color='b')
    plt.plot(cost_constraints, regrets2, color='r')
    axes = plt.gca()
    axes.set_xlim([np.amin(cost_constraints), np.amax(cost_constraints)])
    axes.set_ylim([np.amin(regrets),np.amax(regrets2)])
    plt.show()
