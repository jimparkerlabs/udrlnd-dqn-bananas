import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
GAMMA = 0.99            # discount factor
# LR = 5e-4               # learning rate
# UPDATE_EVERY = 4        # how often to update the network
# BATCH_SIZE = 64         # minibatch size
# TAU = 1e-3              # for soft update of target parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, state_size, action_size, seed, learning_config=None):
        """
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        if learning_config:
            fc1, fc2, fc3, bs, te, ue, tau, lr, b, double = learning_config
            self.batch_size = bs
            self.train_every = te
            self.update_every = ue
            self.tau = tau
            self.b = b
            self.double = double
        else:
            fc1 = int(np.power(2, np.ceil(np.log2(state_size * 4))))  # at least 4 per input, and a power of 2
            fc2 = fc1 * 2
            fc3 = fc2 * 2
            hidden_units = fc1 + fc2 + fc3
            self.batch_size = hidden_units * 8
            self.train_every = int(self.batch_size / 4)
            self.update_every = 1
            self.tau = 0.001
            self.b = None
            self.double = False
            lr = 5e-4

        self.qnetwork_local = QNetwork(state_size, action_size, seed, fc1, fc2, fc3).to(device)
        self.qnetwork_local.apply(self.initialize_weights)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.qnetwork_target = QNetwork(state_size, action_size, seed, fc1, fc2, fc3).to(device)
        # self.qnetwork_local.apply(self.initialize_weights)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, self.batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.u_step = 0

    def initialize_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    def step(self, state, action, reward, next_state, done):
        if self.b:
            # calculate error so we can save it with experience
            next = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
            this = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                predicted_next_q = np.max(self.qnetwork_local(next).cpu().data.numpy())
                predicted_this_q = np.max(self.qnetwork_local(this).cpu().data.numpy())
            self.qnetwork_local.train()

            # calculate error
            error = np.abs(predicted_this_q - (reward + predicted_next_q))
        else:
            error = 1

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, error)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.train_every

        if self.t_step == 0:
            self.u_step = (self.u_step + 1) % self.update_every
            # learn from your experiences (if you have enough)
            experiences = self.memory.sample(self.b)  # TODO: put b on a schedule (small -> 1)
            if experiences:
                self.learn(experiences, GAMMA, n_steps=1, update=(self.u_step == 0))

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, n_steps=1, update=True):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # print('!', end='')
        keys, states, actions, rewards, next_states, dones, w = experiences
        keys.requires_grad = False

        for _ in range(n_steps):
            if self.double:
                # Double DQN:
                # Get next action using the local network
                next_action_local = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)

                # get the Q-values for those actions from the target network
                Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_action_local)
            else:
                # vanilla DQN
                # Get max predicted Q values (for next states) from target model
                next_action_target = self.qnetwork_target(next_states).detach().max(1)[1].unsqueeze(1)
                Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_action_target)

            # Compute Q targets for current states
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states).gather(1, actions)

            # Compute loss
            if w is None:
                loss = F.mse_loss(Q_expected, Q_targets)  # d(Qe-Qt)**2 = 2 * (Qe-Qt) * d Qe
            else:
                # update errors in memory
                new_error = (Q_targets - Q_expected).detach()
                self.memory.update(keys, torch.abs(new_error))

                loss = F.mse_loss(torch.sqrt(w) * Q_expected, torch.sqrt(w) * Q_targets)  # d(wQe-wQt)**2 = 2 * w * (Qe-Qt) * w * d Qe

            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print(f"average loss: {loss.mean()}")

        # ------------------- update target network ------------------- #
        if update:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        # TODO: specify whether to use prioritized sampling
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["key", "state", "action", "reward", "next_state", "done", "error"])
        self.seed = random.seed(seed)
        self.key = 0
    
    def add(self, state, action, reward, next_state, done, error):
        """Add a new experience to memory."""
        # TODO: this thing doesn't seem to care about oveflowing the memory.
        self.key += 1
        e = self.experience(self.key, state, action, reward, next_state, done, error)
        self.memory.append(e)
    
    def sample(self, b=0):
        """Randomly sample a batch of experiences from memory."""
        N = len(self.memory)
        if N < self.batch_size:
            return None

        if b is None:
            b = 0

        if b:
            p = np.power([e.error for e in self.memory], b)
            sum_p = np.sum(p)
            p = p / sum_p
            experiences = random.choices(self.memory, weights=p, k=self.batch_size)  # with replacement (!!!)
        else:
            experiences = random.sample(self.memory, k=self.batch_size)  # without replacement

        keys = torch.from_numpy(np.vstack([e.key for e in experiences if e is not None])).long().to(device)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        if b:
            w = np.power(N * np.power([e.error for e in experiences if e is not None], b) / sum_p, -b)
            w = torch.from_numpy(w / np.max(w)).float().to(device)
        else:
            w = None

        return keys, states, actions, rewards, next_states, dones, w

    def update(self, keys, errors):
        for m in range(len(self.memory)):
            if self.memory[m].key in keys:
                self.add(
                    self.memory[m].state,
                    self.memory[m].action,
                    self.memory[m].reward,
                    self.memory[m].next_state,
                    self.memory[m].done,
                    errors[(keys.squeeze() == self.memory[m].key).nonzero()[0].item()]
                )
                self.memory.remove(self.memory[m])

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
