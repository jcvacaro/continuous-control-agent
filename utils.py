import random
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_scores(filename, scores, label='Score'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel(label)
    plt.xlabel('Episode #')
    plt.savefig(filename, transparent=False)
    plt.close()

def torch_reverse(tensor, axis=0):
    idx = [i for i in range(tensor.size(axis)-1, -1, -1)]
    idx = torch.LongTensor(idx).to(device)
    return tensor.index_select(axis, idx)

def future_rewards(rewards, axis=0):
    return torch_reverse(torch_reverse(rewards, axis=axis).cumsum(axis), axis=axis)

def normalize_rewards(rewards, axis=0):
    return (rewards - rewards.mean().float()) / (rewards.std().float() + 1.0e-10)
