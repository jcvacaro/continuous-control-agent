import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, seed, memory, batch_size, lr_actor, gamma, tau, weight_decay, update_network_steps, sgd_epoch, checkpoint_suffix):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            memory (ReplayBuffer): The replay buffer for storing xperiences
            batch_size (int): Number of experiences to sample from the memory
            lr_actor (float): The learning rate for the actor
            gamma (float): The reward discount factor
            tau (float): For soft update of target parameters
            weight_decay (float): The weight decay
            update_network_steps (int): How often to update the network
            sgd_epoch (int): Number of iterations for each network update
            checkpoint_suffix (string): The string suffix for saving checkpoint files
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.memory = memory
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.gamma = gamma
        self.tau = tau
        self.weight_decay = weight_decay
        self.update_network_steps = update_network_steps
        self.sgd_epoch = sgd_epoch
        self.n_step = 0
        
        # checkpoint
        self.checkpoint_suffix = checkpoint_suffix
        self.actor_loss_episodes = []
        self.actor_loss = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Noise process
        self.noise = OUNoise(action_size, seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(state.shape[0]):
            self.memory.add(state[i], action[i], reward[i], next_state[i], done[i])
         
        # learn every n steps
        self.n_step = (self.n_step + 1) % self.update_network_steps
        if self.n_step == 0 or if np.any(dones):
            for i in range(self.sgd_epoch):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()
        self.memory.reset()
        self.n_step = 0

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, _ = experiences
        
        L = -pong_utils.clipped_surrogate(policy, old_probs, states, actions, rewards,
                                          epsilon=epsilon, beta=beta)
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        del L

    def clipped_surrogate(old_probs, states, actions, rewards, discount=0.995, epsilon=0.1, beta=0.01):
        discount = discount ** torch.arange(rewards.shape[0])
        rewards = rewards * discount
        
        # convert rewards to future rewards
        rewards_future = rewards[::-1].cumsum(0)[::-1]
        mean = rewards_future.mean(1)
        std = rewards_future.std(1) + 1.0e-10
        rewards_normalized = (rewards_future - mean) / std
        
        # convert everything into pytorch tensors and move to gpu if available
        actions = torch.tensor(actions, dtype=torch.int8, device=device)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

        # convert states to policy (or probability)
        new_probs = states_to_prob(policy, states)
        new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs)
        
        # ratio for clipping
        ratio = new_probs / old_probs

        # clipped function
        clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
            (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

        
        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        return torch.mean(clipped_surrogate + beta*entropy)

    def checkpoint(self):
        """Save internal information in memory for later checkpointing"""
        self.actor_loss_episodes.append(self.actor_loss)

    def save_checkpoint(self):
        """Persist checkpoint information"""
        # the history loss
        utils.plot_scores("actor_loss_" + self.checkpoint_suffix + ".png", self.actor_loss_episodes, label="loss")
        
        # network
        torch.save(self.actor_local.state_dict(), "actor_" + self.checkpoint_suffix + ".pth")
        
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
