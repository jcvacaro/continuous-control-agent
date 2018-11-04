import random
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

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
