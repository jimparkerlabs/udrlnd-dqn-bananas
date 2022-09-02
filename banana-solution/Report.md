# Banana Report

The RL algorithm used was Double-DQN with experience replay and a fixed Target network updated periodically.  

The neural network with 2 hidden layers, each with number of neurons = .  The network consisted of 2 fully connected hidden Linear layers with state_size * 2 (= 74) neurons each and ReLU activations.  The network weights were initialized using xavier_uniform initialization and trained at a learning rate on 0.0005.

Because this is Double DQN, there were 2 identical networks used.  The online network was trained on a random selection of 64 prior experiences every 4 time steps, and the weights thus learned were copied to the Target network every 4 training cycles (every 16 time steps).  These weights were not transferred completely, the update percentage was 0.001.  Also, this Target network was used to evaluate the Q errors while the online network was used to choose the actions (this is the "double" part of Double DQN). 

### Plot of Rewards

![](output.png)
Environment solved in 682 episodes!	Average Score: 13.01

### Ideas for Future Work

I actually implemented Prioritized recall, but the implementation was very inefficient, so I abandoned that approach.  An obvious place to improve this learner would be to fix the implementation of Prioritized Experience Replay.

I considered but did not attempt a Dueling DQN, but that would have been my next improvement after the Prioritized Experience Replay.
