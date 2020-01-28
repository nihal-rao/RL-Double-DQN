# RL-DQN

**Using Double Deep Q Networks with experience replay to solve Cartpole v0 in just 184 steps.**
**Implemented using tensorflow 2.**

# Double Deep Q Networks and Experience Replay:

* Double Q Learning was first introduced [in this paper by DeepMind](https://arxiv.org/pdf/1509.06461.pdf). 
* The main idea is to use two estimators instead of just one, thereby decoupling action selection from target estimation. 
* Experience replay is used to reduce correlation between training samples for the agent.
* This also improves data efficiency as previous experiences are used to train the agent.
* Instead of hard updating(copying the primary network's parameters) the target network, Polyak averaging is used to "blend" the target network with the primary network.

# Implementation Details:
* CartPole-v0 defines **"solving" as getting average reward of 195.0 over 100 consecutive trials.**
* Only the primary network is trained, the target network is just "soft" updated using the primary network every episode.
* I find that a batch size of 16 is faster than the usual 32.
* The use of a learning rate scheduler is key to ensuring convergence. Withou this we see that though the maximum reward is obtained the task is never solved :

