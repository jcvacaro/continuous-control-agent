import random
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_checkpoint(scores_episodes, actor_loss_episodes, critic_loss_episodes, scores_window):
    plot_rewards("reward_history_plot_" + args.checkpoint_suffix + ".png", scores_episodes)
    plot_rewards("actor_loss_" + args.checkpoint_suffix + ".png", actor_loss_episodes)
    plot_rewards("critic_loss_" + args.checkpoint_suffix + ".png", critic_loss_episodes)
    plot_rewards("reward_plot_" + args.checkpoint_suffix + ".png", scores_window)

def plot_scores(filename, scores, label='Score'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel(label)
    plt.xlabel('Episode #')
    plt.savefig(filename, transparent=False)
    plt.close()

def torch_inv(tensor, axis=0):
    idx = [i for i in range(tensor.size(axis)-1, -1, -1)]
    idx = torch.LongTensor(idx).to(device)
    return tensor.index_select(axis, idx)

def future_rewards(rewards, axis=0):
    return torch_inv(torch_inv(rewards, axis=axis).cumsum(axis), axis=axis)

def normalize_rewards(rewards, axis=0):
    return (rewards - rewards.mean(axis).float()) / (rewards.std(axis).float() + 1.0e-10)