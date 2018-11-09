import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import PPOActor
import utils

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, seed, memory, batch_size, lr_actor, gamma, eps, eps_decay, beta, beta_decay, weight_decay, update_network_steps, sgd_epoch, checkpoint_suffix):
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
            eps (float): The PPO epsilon clipping parameter
            eps_decay (float): Epsilon decay value
            beta (float): The PPO regulation term for exploration
            beta_decay (float): The beta decay value
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
        self.eps = eps
        self.eps_decay = eps_decay
        self.beta = beta
        self.beta_decay = beta_decay
        self.weight_decay = weight_decay
        self.update_network_steps = update_network_steps
        self.sgd_epoch = sgd_epoch
        self.n_step = 0
        
        # checkpoint
        self.checkpoint_suffix = checkpoint_suffix
        self.actor_loss_episodes = []
        self.actor_loss = 0

        # Actor Network (w/ Target Network)
        self.actor = PPOActor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=weight_decay)

    def step(self, state, action, action_prob, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, action_prob, reward, next_state, done)
         
        # learn every n steps
        self.n_step = (self.n_step + 1) % self.update_network_steps
        if self.n_step == 0 or np.any(done):
            experiences = self.memory.sample()  # retrieve trajectories
            for i in range(self.sgd_epoch):
                self.learn(experiences, self.gamma)
            self.eps *= self.eps_decay          # the PPO clipping parameter reduces as time goes on
            self.beta *= self.beta_decay        # the regulation term reduces exploration in later runs
            self.memory.reset()                 # reset memory

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action, action_prob = self.actor(state)
        self.actor.train()
        return np.clip(action.cpu().data.numpy(), -1, 1), action_prob.cpu().data.numpy()

    def reset(self):
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
        states, actions, action_probs, rewards, next_states, dones = experiences
        
        L = -self.clipped_surrogate(states, actions, action_probs, rewards)
        self.actor_loss = L

        self.actor_optimizer.zero_grad()
        L.backward()
        #torch.nn.utils.clip_grad_norm(self.actor.parameters(), 5)
        self.actor_optimizer.step()

    def clipped_surrogate(self, states, actions, action_probs, rewards):
        discount = self.gamma ** torch.arange(len(rewards)).to(device)  # compute the discounts
        rewards = rewards * discount.float().view(-1, 1, 1)             # discounted rewards
        rewards = utils.future_rewards(rewards)                         # convert rewards to future rewards
        rewards = utils.normalize_rewards(rewards)                      # normalize rewards

        # convert states to policy (or probability)
        new_actions, new_action_probs = self.states_to_prob(states)
        
        # Ratio for clipping. We are dealing with log probabilities, hence the minus, not division.
        ratio = (new_action_probs - action_probs).exp()

        # clipped function
        clip = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
        clipped_surrogate = torch.min(ratio * rewards, clip * rewards)

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -new_action_probs.exp() * new_action_probs

        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        return torch.mean(clipped_surrogate + self.beta * entropy)

    def states_to_prob(self, states):
        """Convert states to probability, passing through the policy"""
        T, A, state_size = states.shape                             # T=time steps, A=number of agents
        flat_states = states.view(-1, state_size)                   # from [T,A,size] tp [T*A,size]
        actions, log_probs = self.actor(flat_states)                # invoke the policy
        return map(lambda x: x.view(T, A, x.shape[-1]), (actions, log_probs))

    def checkpoint(self):
        """Save internal information in memory for later checkpointing"""
        self.actor_loss_episodes.append(self.actor_loss)

    def save_checkpoint(self):
        """Persist checkpoint information"""
        # the history loss
        utils.plot_scores(self.checkpoint_suffix + "_actor_loss.png", self.actor_loss_episodes, label="loss")
        
        # network
        torch.save(self.actor.state_dict(), self.checkpoint_suffix + "_actor.pth")
        
    def load_checkpoint(self):
        """Restore checkpoint information"""
        self.actor.load_state_dict(torch.load(self.checkpoint_suffix + "_actor.pth"))
