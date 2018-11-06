from unityagents import UnityEnvironment
import random
import torch
import numpy as np
from collections import deque
import os
import argparse
import matplotlib.pyplot as plt

import utils
from memory import DeterministicReplayBuffer, UniformReplayBuffer
import ddpg_agent as ddpg
import ppo_agent as ppo

# Arguments Parsing Settings
parser = argparse.ArgumentParser(description="DQN Reinforcement Learning Agent")

parser.add_argument('--seed', help="Seed for random number generation", type=int, default=3)
parser.add_argument('--env', help="The environment path", default="Reacher_1/Reacher.app")
parser.add_argument('--checkpoint_suffix', help="The string suffix for saving checkpoint files", default="default")

# training/testing flags
parser.add_argument('--train', help="train or test (flag)", action="store_true")
parser.add_argument('--algorithm', choices=["ddpg", "ppo"], help="The algorithm", default="ddpg")
parser.add_argument('--test_episodes', help="The number of episodes for testing", type=int, default=3)
parser.add_argument('--train_episodes', help="The number of episodes for training", type=int, default=650)
parser.add_argument('--batch_size', help="The mini batch size", type=int, default=512)
parser.add_argument('--gamma', help="The reward discount factor", type=float, default=0.99)
parser.add_argument('--lr_actor', help="The learning rate for the actor", type=float, default=1e-4)
parser.add_argument('--lr_critic', help="The learning rate for the critic", type=float, default=1e-3)
parser.add_argument('--clip_critic', help="The clip value for updating grads", type=float, default=1)
parser.add_argument('--tau', help="For soft update of target parameters", type=float, default=1e-3)
parser.add_argument('--weight_decay', help="The weight decay", type=float, default=1e-4)
parser.add_argument('--update_network_steps', help="How often to update the network", type=int, default=30)
parser.add_argument('--sgd_epoch', help="Number of iterations for each network update", type=int, default=10)
parser.add_argument('--eps', help="The PPO epsilon clipping parameter", type=float, default=0.1)
parser.add_argument('--eps_decay', help="Epsilon decay value", type=float, default=0.999)
parser.add_argument('--beta', help="The PPO regulation term for exploration", type=float, default=0.01)
parser.add_argument('--beta_decay', help="The beta decay value", type=float, default=0.995)

# replay memory 
parser.add_argument('--buffer_type', choices=["deterministic", "uniform"], help="The replay buffer type", default="uniform")
parser.add_argument('--buffer_size', help="The replay buffer size", type=int, default=int(1e5))

# model
parser.add_argument('--network', choices=["ddpg_linear1", "ddpg_linear2"], help="The neural network model", default="ddpg_linear1")

def create_environment():
    env = UnityEnvironment(file_name=args.env, no_graphics=True)
    #env = UnityEnvironment(file_name="Reacher_Env/Reacher.app", docker_training=True, no_graphics=False)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    return env, brain, brain_name, num_agents, action_size, state_size

def train(agent, env, brain, brain_name, num_agents, n_episodes):
    scores_episodes = deque(maxlen=n_episodes)                  # The score history over all episodes
    scores_window = deque(maxlen=100)                           # last 100 scores

    for i_episode in range(1, n_episodes+1):
        agent.reset()                                           # reset the agent
        env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
        states = env_info.vector_observations                   # get initial observation
        scores = np.zeros(num_agents)                           # initialize the score (for each agent)

        while True:
            # agent chooses an action
            actions, action_probs = agent.act(states)
            
            # interact with the environment
            env_info = env.step(actions)[brain_name]            # send all actions to tne environment
            next_states = env_info.vector_observations          # get next state (for each agent)
            rewards = env_info.rewards                          # get reward (for each agent)
            dones = env_info.local_done                         # see if episode finished
            
            # agent learns with the new experience
            agent.step(np.asarray(states), 
                       np.asarray(actions), 
                       np.asarray(action_probs),
                       np.asarray(rewards)[:, np.newaxis],
                       np.asarray(next_states), 
                       np.asarray(dones)[:, np.newaxis])

            scores += rewards                                   # update the score (for each agent)
            states = next_states                                # roll over states to next time step
            if np.any(dones):                                   # exit loop if episode finished
                break

        # checkpoint 
        score = np.mean(scores)
        scores_episodes.append(score)                           # save most recent score in the history
        scores_window.append(score)                             # save most recent score
        agent.checkpoint()                                      # agent checkpoint
        
        # verify if the goal has been achieved
        score_window = np.mean(scores_window)                   # the mean of the last 100 episodes
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score_window), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score_window))
            save_checkpoint(agent, scores_episodes, scores_window)
        if score_window >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, score_window))
            break

    # plot score history
    save_checkpoint(agent, scores_episodes, scores_window)
  
def save_checkpoint(agent, scores_episodes, scores_window):
    utils.plot_scores("reward_history_plot_" + args.checkpoint_suffix + ".png", scores_episodes)
    utils.plot_scores("reward_plot_" + args.checkpoint_suffix + ".png", scores_window)
    agent.save_checkpoint()

if __name__ == '__main__':
    args = parser.parse_args()
    
    # environment
    env, brain, brain_name, num_agents, action_size, state_size = create_environment()

    # replay memory
    if args.buffer_type == "uniform":
        memory = UniformReplayBuffer(action_size=action_size, 
                                     state_size=state_size, 
                                     buffer_size=args.buffer_size,
                                     batch_size=args.batch_size,
                                     seed=args.seed)
    else:
        memory = DeterministicReplayBuffer(action_size=action_size, 
                                           state_size=state_size, 
                                           buffer_size=args.buffer_size)

    # agent
    if args.algorithm == "ddpg":
        agent = ddpg.Agent(state_size=state_size, 
                           action_size=action_size, 
                           seed=args.seed,
                           batch_size=args.batch_size,
                           memory=memory,
                           lr_actor=args.lr_actor,
                           lr_critic=args.lr_critic,
                           clip_critic=args.clip_critic,
                           gamma=args.gamma,
                           tau=args.tau,
                           weight_decay=args.weight_decay,
                           update_network_steps=args.update_network_steps,
                           sgd_epoch=args.sgd_epoch,
                           checkpoint_suffix=args.checkpoint_suffix)
    else:
        agent = ppo.Agent(state_size=state_size, 
                          action_size=action_size, 
                          seed=args.seed,
                          batch_size=args.batch_size,
                          memory=memory,
                          lr_actor=args.lr_actor,
                          gamma=args.gamma,
                          eps=args.eps,
                          eps_decay=args.eps_decay,
                          beta=args.beta,
                          beta_decay=args.beta_decay,
                          weight_decay=args.weight_decay,
                          update_network_steps=args.update_network_steps,
                          sgd_epoch=args.sgd_epoch,
                          checkpoint_suffix=args.checkpoint_suffix)

    if args.train:
        train(agent, env, brain, brain_name, num_agents, n_episodes=args.train_episodes)

#    else:
#        test(agent, env, brain, brain_name, n_episodes=args.test_episodes)

    env.close()
