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

### The replay Buffer

`memory.py` holds the implementation for the memory buffers strategies. The DDPG algorithm uses a uniform sampling buffer with the objective of training the model by first storing experiences in the buffer, and then replaying a batch of experiences from it in a subsequent step. The expectation is to reduce the correlation of such observations, which leads to a more stable training procedure. The implementation of this strategy is defined in the ReplayBuffer class.

### The neural network

`model.py` implements the neural network architecture for the actor and the critic. Both models are very similar, consisting of 3 Multilayer perceptron (MLP) layers. Each layer uses the RELU action function, except the last one, which has dimension equivalent to the number of actions and applies the `tanh` activation function. That is because `tanh` outputs values between -1 and 1, exactly the continuous action space needed for solving the target environment. The table below shows the complete network model configuration:

- Actor

| Layer |  type  | Input | Output | Activation |
| ----- | ------ | ----- | ------ | ---------- |
| 1     | linear | 33    | 400    | RELU       |
| 2     | linear | 400   | 300    | RELU       |
| 3     | linear | 300   | 4      | TANH       |

- Critic: The second layer also includes the action, Q(s, a)

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



## References

- [Human-Level Control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
