[//]: # (Image References)

[image1]: https://drive.google.com/uc?id=1ET-CutV_uBdrM6w_u4Sw7MOkkFSz4juE "Single Agent Training"
[image2]: https://drive.google.com/uc?id=1qYKBiuLU_IXYDtzVaDLhK0pjVJeIm_gs "Multi-Agent Training"

# Report

## Learning Algorithm

It is possible to identify four distinct components in DDPG-based algorithms: (1) the training loop, agent--environment interaction; (2) the DDPG algorithm itself; (3) the particular replay buffer strategy; and (4) the neural network used to generate the policy and the critic Q-values. Although DDPG is presented as an actor critic reinforcement learning method, it has many similarities with the value-base DQN algorithm.

### The training loop

`main.py` contains the traditional training loop:

- initialize the environment
- for each episode
    - get the initial state from the environment
    - for each step in the current episode
        - get the next action from the agent
        - get the next state and reward from the environment
        - the agent observes the new experience and improve its model
        - verifiy if the agent achieved the goal

### The DDPG algorithm
    
`ddpg_agent.py` contains theDDPG algorithm. Each time the training loop observes a new experience (state, action, reward, next state, done) the method `step` is invoked. The experience is stored into the replay buffer, and if the buffer contains at least batch size entries, a new batch of experience is sampled. Note that the same code works for the single and multi-agent environments. That is because each experience are added to the replay buffer independently from the agent that generated it.

There are 4 networks involved in DDPG: the local actor, the local critic, the target actor, and the target critic. The `learn` method  trains such networks in three steps:

1. The critic loss: First, predicted next-state actions and Q values from the target models. The next action is obtained from the target actor, and the next Q-value from the target critic model. Then, the loss is obtained by the TD (temporal difference) between the just calculated next Q-value, and the current Q-value from the local critic network. 

2. The actor loss: We predict the action from the local actor model based on the current experience state, and then the Q-value from the local critic. Then, the critic Q-value is the actual loss for the actor, that is, the critic guides the direction of the actor gradient updates.

3. Soft updates: Instead of performing a huge update every n steps, DDPG slowly updates the target networks based on the local networks every step. Remember that the local networks are the most updated because those are the ones being trained. The target networks are slowly updated to maintain stability during learning.

Training the agent on the target environment was really involving. The following techniques contributed to the stability of the algorithm significantly:

- Normalizing rewards: Looking at the reward plot at the begining of the journey, it was clear that the values were very noisy over time. Applying normalization made the values more standard when dealing with different magnitudes, and also reduced extreme values.
- Increasing the batch size: Going from 128 to 512 also reduced the noise in the actor and critic loss. That is probably because a more significant sampling is performed from the total entries in the replay memory.
- Clipping the critic gradients: It is very clear the relationship between the critic and the actor loss, and of course that impacts the rewards. So, since the critic guides the actor gradient updates, reducing the variance of the critic improves the whole system. The goal is to use a clip function in the gradients to eliminate extreme updates in the critic network.
- Choosing the weight decay value for the optimizer: Here the actor loss plot helped a lot. The loss started very well decreasing over time, but after a certain number of episodes it diverged completely by increasing its value. The weight decay was criticaly important for solving this problem. By decreasing the learning rate during the optimization step probably contributed for the network to reach a better local minimum. 

### The replay Buffer

`memory.py` holds the implementation for the memory buffers strategies. The DDPG algorithm uses a uniform sampling buffer with the objective of training the model by first storing experiences in the buffer, and then replaying a batch of experiences from it in a subsequent step. The expectation is to reduce the correlation of such observations, which leads to a more stable training procedure. The implementation of this strategy is defined in the ReplayBuffer class.

### The neural network

`model.py` implements the neural network architecture for the actor and the critic. Both models are very similar, consisting of 3 Multilayer perceptron (MLP) layers. Each layer uses the RELU action function, except the last one, which has dimension equivalent to the number of actions and applies the `tanh` activation function. That is because `tanh` outputs values between -1 and 1, exactly the continuous action space needed for solving the target environment. The table below shows the complete network model configuration:

#### Actor

The actor represents the policy.

| Layer |  type  | Input | Output | Activation |
| ----- | ------ | ----- | ------ | ---------- |
| 1     | linear | 33    | 400    | RELU       |
| 2     | linear | 400   | 300    | RELU       |
| 3     | linear | 300   | 4      | TANH       |

#### Critic

The critic represents the Q-value function Q(s, a). The action is incorporated into the second layer of the network.

| Layer |  type  | Input | Output | Activation |
| ----- | ------ | ----- | ------ | ---------- |
| 1     | linear | 33    | 400    | RELU       |
| 2     | linear | 404   | 300    | RELU       |
| 3     | linear | 300   | 4      | TANH       |

## Results

Both the single and multi-agent environments are solved using the DDPG algorithm. For the single agent, the goal is achieve in 606 episodes. The following graph shows the reward progression of the last 100 episodes where the agent achieved +30 points.

![Single Agent Training][image1]

The complete configuration is shown in the table below.

| Parameter       | Description                                        | Value    |
| --------------- | -------------------------------------------------- | -------- |
| algorithm       | The actor-critic algorithm                         | DDPG     |
| replay_buffer   | The replay buffer strategy                         | uniform  |
| buffer_size     | The replay buffer size                             | 1e5      |
| batch_size      | The batch size                                     | 512      |
| gamma           | The reward discount factor                         | 0.99     |
| tau             | For soft update of target parameters               | 1e-3     |
| lr_actor        | The learning rate                                  | 1e-4     |
| lr_critic       | The learning rate                                  | 1e-3     |
| clip_critic     | Clip the critic gradients during training          | 1        |
| weight_decay    | The decay value for the Adam optmizer algorithm    | 1e-4     |
| update_steps    | Number of steps to apply the learning procedure    | 30       |
| sgd_epoch       | Number of training iterations for each learning    | 10       |

For the multi-agent environment, the goal is achieved in 202 episodes. The following graph shows the reward progression of the last 100 episodes where the agent achieved +30 points. Here, the weight decay, and choosing a more agressive updating step for updating the network impacted the performance significantly.

![Multi-Agent Training][image2]

The complete configuration is shown in the table below.

| Parameter       | Description                                        | Value    |
| --------------- | -------------------------------------------------- | -------- |
| algorithm       | The actor-critic algorithm                         | DDPG     |
| replay_buffer   | The replay buffer strategy                         | uniform  |
| buffer_size     | The replay buffer size                             | 1e5      |
| batch_size      | The batch size                                     | 512      |
| gamma           | The reward discount factor                         | 0.99     |
| tau             | For soft update of target parameters               | 1e-3     |
| lr_actor        | The learning rate                                  | 1e-4     |
| lr_critic       | The learning rate                                  | 1e-3     |
| clip_critic     | Clip the critic gradients during training          | 1        |
| weight_decay    | The decay value for the Adam optmizer algorithm    | 25e-5    |
| update_steps    | Number of steps to apply the learning procedure    | 20       |
| sgd_epoch       | Number of training iterations for each learning    | 10       |

## Ideas for future work

- Distribute computation across multiple machines: The multi-agent environment converged much faster than the single one. It will be interesting to see how it behaves when multiple machines are involved in the learning process. Another aspect is that agents will work asynchronously.
- Compare different algorithms: The current DDPG implementation actually solves both environments. However, there are other strategies that could be applied such as PPO, D4PG, A3C. What are the differences? When to use one or the other? This repository contains an experimental version for PPO, and I will be adding support for other algorithms soon in order to perform benchmarks.

## References

- [DDPG](https://arxiv.org/abs/1509.02971)
- [PPO](https://arxiv.org/pdf/1707.06347.pdf)
- [A3C](https://arxiv.org/pdf/1602.01783.pdf)
- [D4PG](https://openreview.net/pdf?id=SyZipzbCb)
