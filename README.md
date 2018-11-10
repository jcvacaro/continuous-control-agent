# Continuous Control Reinforcement Learning Agents

This repository contains implementations in Pytorch of DDPG (Deep Deterministic Policy Gradients) and experimental support of PPO (Proximal Policy Optimization) for solving continuous tasks. The environment is defined in the Unity engine, and the communication is based on the Unity ML Agents API. See the Report.md file for in depth details about the algorithms used and the organization of the source code.

## Goal

The target environment is the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher).

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

There are two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

### Solving the Environment

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically:

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

## Getting Started

The source code is implemented in Python 3x, and uses PyTorch as the Machine Learning framework. 

1. Install PyTorch
    - Windows: [Anaconda](https://conda.io/docs/user-guide/install/windows.html), [PyTorch](https://pytorch.org/get-started/locally/)
    - Linux: [Anaconda](https://conda.io/docs/user-guide/install/linux.html), [PyTorch](https://pytorch.org/get-started/locally/)
    - Docker: See the Dockerfile for instructions about how to generate the image
2. Download the environment from one of the links below. You need only select the environment that matches your operating system:
    - Single Agent: [Linux](https://drive.google.com/uc?id=1RDmEUl8OxibLfMAHphY9LcKg-f5Yra-R), [Windows 64-bit](https://drive.google.com/uc?id=1QRSQECQf95Qh1_OdTPau9D6WCiwLGKkD)
    - Multi-Agent: [Linux](https://drive.google.com/uc?id=1prC-ZHLWEKcoMjQllx6HJyTu7rclzDM4), [Windows 64-bit](https://drive.google.com/uc?id=1wa78vhgi370N8JcJ9A4j9k0QBKIFdA4x)
3. To test the agent with a pre-trained network, download one of the model checkpoints:
    - Single Agent: [actor](https://drive.google.com/uc?id=1OuutszmDw4-Cp--1GCBpXecy-mli4gQ-), [critic](https://drive.google.com/uc?id=1VfH2mZYHxhVeMd-lGv3BrBcvyCxzce2n)
    - Multi-Agent: [actor](https://drive.google.com/uc?id=1Ix9iZ4ja1KXs1_IQd_oHIE6KGHCXzI2w), [critic](https://drive.google.com/uc?id=1XgzpuK3eR59EgMqbHhoEz5Fry0WkaPIp)
4. Place the file(s) in the repository folder, and unzip (or decompress) the file(s).

## Instructions

The main.py is the application entry point. To show all available options:

```bash
python main.py --help
```

To train the agent:

```bash
python main.py --train \
    --train_episodes=800 \
    --checkpoint_prefix=reacher_ddpg_single \
    --env=Reacher_1/Reacher.app \
    --batch_size=512 \
    --update_network_steps=30 \
    --sgd_epoch=10 \
    --clip_critic=1 \
    --weight_decay=1e-4
```

Note: By default, the agent uses the DDPG algorithm with the single agent environment. The training runs 800 episodes, and saves the model checkpoint in the current directory if the goal is achieved.

To select the multi-agent environment, issue the following command:

```bash
python main.py --train \
    --train_episodes=400 \
     --checkpoint_prefix=reacher_ddpg_multi_agent \
     --env=Reacher_20/Reacher.app \
    --batch_size=512 \
    --update_network_steps=20 \
    --sgd_epoch=10 \
    --clip_critic=1 \
    --weight_decay=25e-5
```

To test the agent using a model checkpoint:

```bash
python main.py \
    --test_episodes=3 \
    --checkpoint_prefix=reacher_ddpg_multi_agent \
    --env=Reacher_20/Reacher.app
```

In addition, many hyper parameters can be customized such as the learning rate, the reward discount factor gamma. Check the --help for all available options.
