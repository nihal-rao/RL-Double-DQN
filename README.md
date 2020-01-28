# RL-Double DQN

**Using Double Deep Q Networks with experience replay to solve [Cartpole v0](https://github.com/openai/gym/wiki/CartPole-v0) in just 184 episodes.**
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
* The use of learning rate decay is key to ensuring convergence. Without this we see that though the maximum reward is obtained the task is never solved :

![alt text](https://github.com/nihal-rao/RL-DQN/blob/master/images/no_lr_decay.png "Reward v Episodes, without LR decay")
* The initial value of epsilon is set as 0.5. This decays as the agent is trained, with its minimum value being 0.01.
* The Agent class from agent.py contains the implementation, training is done on cartpole.ipynb (on Google Colab).

# Results:
* Cartpole v0 is solved after 184 training episodes.
* Graphs of reward and mean average of reward over 100 consecutive trials :

![alt text](https://github.com/nihal-rao/RL-DQN/blob/master/images/results.png "Results")
* The agent beng tested, with epsilon set to zero (it obtains the maximum possible reward of 200) :
![alt text](https://github.com/nihal-rao/RL-DQN/blob/master/cartpole.gif)
