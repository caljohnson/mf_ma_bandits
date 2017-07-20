# Multifidelity Extension of the Multi-Armed Bandit

Extends multi-armed bandit classes to include multi-fidelity versions, and implements the MF-UCB algorithm as detailed in [Kandasamy et. al.](https://arxiv.org/abs/1610.09726)

---

Multi-fidelity multi-armed bandits have several 'versions' of the same arms, with increasing fidelities and costs.  Below is an example of the mean rewards per arm and fidelity of a 5-armed, 3-fidelity Gaussian-reward bandit.  Fidelity 0 are the low fidelity, high bias/variance observations of the bandit arms, and the error bars denote the maximum distance away from the Fidelity 2 mean reward.
![alt text](https://github.com/drruumms/bandits/blob/master/mf_bandits/mf_rewards_case3.png "MF-MA Bandit Rewards")

In this case, the lower fidelity arms contain enough information about the high fidelity arms for a MF-UCB agent to leverage, so the overall regret is lower compared to a UCB agent playing only the high fidelity arms.  The lower cost of the lower fidelity arms allows the MF-UCB agent to explore suboptimal arms and rule them out at a lower cost than a UCB agent.
![alt text](https://github.com/drruumms/bandits/blob/master/mf_bandits/mf_plays_case3.png "MF-UCB vs. UCB Plays")
![alt text](https://github.com/drruumms/bandits/blob/master/mf_bandits/mf_regret_Case3.png "MF-UCB vs. UCB Regret")

However, I was able to devise an adversarial example that fits the MF-UCB constraints, but performs worse than with the UCB algorithm.
![alt text](https://github.com/drruumms/bandits/blob/master/mf_bandits/mf_rewards_case5.png "MF-MA Bandit Rewards")

In this case, the lower fidelity arms mislead the MF-UCB about which high fidelity arms are optimal, so the overall regret is higher compared to a UCB agent playing only the high fidelity arms.  The lower fidelity information is essentially useless, so the MF-UCB agent wastes resources exploring suboptimal arms and can never catch up to the UCB agent.  By tweaking hyperparameters to disincentive using the low fidelity information, the MF-UCB agent can only perform as well as the UCB agent (by essentially ignoring the low fidelity arms).
![alt text](https://github.com/drruumms/bandits/blob/master/mf_bandits/mf_plays_case5.png "MF-UCB vs. UCB Plays")
![alt text](https://github.com/drruumms/bandits/blob/master/mf_bandits/mf_regret_case5.png "MF-UCB vs. UCB Regret")