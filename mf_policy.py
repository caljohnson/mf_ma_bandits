#mf_policy.py
#multi-fidelity extension of
#Byron Galbraith's multi-armed bandit simulator Policy classes
#source: https://github.com/bgalbraith/bandits/blob/master/bandits/policy.py


import numpy as np


class Policy(object):
    """
    A policy prescribes an action to be taken based on the memory of an agent.
    """
    def __str__(self):
        return 'generic policy'

    def choose(self, agent):
        return 0


class UCBPolicy(Policy):
    """
    The Upper Confidence Bound algorithm (UCB1). It applies an exploration
    factor to the expected value of each arm which can influence a greedy
    selection strategy to more intelligently explore less confident options.

    ONLY LOOKS AT HIGHEST FIDELITY
    """
    def __init__(self, rho, psi_inv):
        self.psi_inv = psi_inv
        self.rho = rho
    
    def __str__(self):
        return 'UCB'

    def choose(self, agent):
        #compute exploration term of UCB
        exploration = self.rho*np.log(agent.t+1)/(agent.action_attempts)
        exploration[np.isinf(exploration)] = 0
        exploration = self.psi_inv(exploration)

        #recall that rows=arms, cols=fidelities, compute total UCB for each arm+fidelity
        q = agent.value_estimates + exploration
        #print(q)
        #get q bound for each arm
        q_arms = q[:,agent.m-1]
        #print(q_arms)
        #pick arm that maximizes arm bounds
        max_arm_index = np.argmax(q_arms)
        #print(max_arm_index)
        
        #play best arm at highest fidelity
        action=[max_arm_index,agent.m-1]
        check = np.where(q == action)[0]
        if check.shape[0] == 0:
            return action
        else:
            return np.random.choice(check)

        #return action

class MF_UCBPolicy(Policy):
    """
    The Multi-Fidelity Upper Confidence Bound algorithm (MF-UCB). It applies an exploration
    factor to the expected value of each arm which can influence a greedy
    selection strategy to more intelligently explore less confident options.
    """
    def __init__(self, rho, psi_inv):
        self.rho = rho
        self.psi_inv = psi_inv

    def __str__(self):
        return 'MF_UCB'

    def choose(self, agent):
        #compute exploration term of UCB
        # exploration = np.power(agent.action_attempts,-1)*self.rho*np.log(agent.t+1)
        exploration = self.rho*np.log(agent.t+1)/(agent.action_attempts)
        exploration[np.isnan(exploration)] = 0
        exploration = self.psi_inv(exploration)

        #recall that rows=arms, cols=fidelities, compute total UCB for each arm+fidelity
        q = agent.value_estimates + exploration + agent.zeta
        #q2 = agent.value_estimates + exploration - agent.zeta
        #get min q bounds across fidelities for each arm
        min_arm_bounds = np.amin(q, axis=1)
        #min_arm_bounds2 = np.amin(q2, axis=1)
        #pick arm that maximizes min arm bounds
        max_arm_index = np.argmax(min_arm_bounds)
        #max_arm_index2 = np.argmax(min_arm_bounds2)
        #pick max of those two
        # if min_arm_bounds[max_arm_index]<min_arm_bounds2[max_arm_index2]:
        #     print("actually using this change")
        #     max_arm_index = max_arm_index2
        #pick lowest, most uncertain fidelity
        action = None
        for m in range(agent.m-1):
            if exploration[max_arm_index, m]>=agent.gamma_fn[m]:
                #print('playing arm ', max_arm_index)
                #print('playing fidelity ', m)
                action=[max_arm_index, m]
                break    
        if action==None:
            #if low fidelities all certain, play at highest fidelity
            action=[max_arm_index,agent.m-1]

        return action