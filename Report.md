[//]: # (Image References)

[image1]: https://drive.google.com/uc?id=1rR6gnP-6y0EnEykJfVZ8CIsKcitOclTx "Trained Agent"

# Report

## Learning Algorithm

It is possible to identify four distinct components in DQN-based algorithms: (1) the training loop, agent--environment interaction; (2) the DQN algorithm flavor; (3) the particular replay buffer strategy; and (4) the neural network used to generate Q-values.

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

### The DQN algorithm
    
`agent.py` contains the DQN/Double DQN algorithms. Each time the training loop observes a new experience (state, action, reward, next state, done) the method `step` is invoked. The experience is stored into the replay buffer, and if the buffer contains at least batch size entries, a new batch of experience is sampled. 

The `learn` method  trains both the local and target networks based on such experiences. For the original DQN algorithm, the next Q-values are 
obtained directly from the target network. Unfortunately, such direct evaluation may lead to overoptimistic value estimates. In the Double DQN algorithm, the action selection and evaluation are performed in different steps. First the action is obtained from the local network, it is the index of the max Q-value. Then, the evaluation of such action is obtained from the  target network. It is the corresponding Q-value of the previously selected action index. For both cases, the reward and discount factor gamma are applied to the target Q-value.

Finally, the TD error is computed as the difference between the target and current Q-values, and the loss is calculated according to the weights generated by the replay buffer strategy.

### The replay Buffer

`memory.py` holds the implementation for the memory buffers strategies. The original DQN algorithm proposes a uniform sampling buffer with the objective of training the model by first storing experiences in the buffer, and then replaying a batch of experiences from it in a subsequent step. The expectation is to reduce the correlation of such observations, which leads to a more stable training procedure. The implementation of this strategy is defined in the ReplayBuffer class.

An interesting strategy is to control which experiences to sample from the buffer in order to maximize the learning. A possible solution is to get  experiences with higher TD error. Considering that higher TD errors provide more aggressive gradients to the network, it would approximate the optimal Q -value function faster. This is called prioritized experience replay, and it is implemented by the PrioritizedReplayBuffer class. 

### The neural network

`model.py` implements the neural network architecture. It consists of 4 Multilayer perceptron (MLP) layers. Each layer uses the RELU action function, except the last one, which has dimension equivalent to the number of actions. Each output unit represents the Q-value for that particular action. The table below shows the complete network model configuration:

| Layer |  type  | Input | Output | Activation |
| ----- | ------ | ----- | ------ | ---------- |
| 1     | linear | 37    | 128    | RELU       |
| 2     | linear | 128   | 64     | RELU       |
| 3     | linear | 64    | 16     | RELU       |
| 4     | linear | 16    | 4      | -          |

This module also provides a Dueling Network Architecture implementation. The proposal is to define a model with two separate streams: one for the value function V(s), and the second one for the advantage function A(a, s). Such architecture performs better because the value function allows that for many states, it is unnecessary to estimate the value of each action choice. The DuelQNetwork class implements minimum modifications to the linear model described above to understand the benefits in practice.

| Layer |  type  | Input | Output | Activation |
| ----- | ------ | ----- | ------ | ---------- |
| 1     | linear | 37    | 128    | RELU       |
| 2     | linear | 128   | 64     | RELU       |
| 3     | linear | 64    | 16     | RELU       |
| 4.1   | A(a,s) | 16    | 4      | -          |
| 4.2   |  V(s)  | 16    | 1      | -          |

## Results

The environment is solved in 478 episodes. The following graph shows the reward progression of the last 100 episodes where the agent achieved +13 points.

![Trained Agent][image1]

The complete configuration is shown in the table below.

| Parameter       | Description                                        | Value    |
| --------------- | -------------------------------------------------- | -------- |
| algorithm       | The DQN flavor                                     | DDQN     |
| replay_buffer   | The replay buffer strategy                         | uniform  |
| buffer_size     | The replay buffer size                             | 1e5      |
| batch_size      | The batch size                                     | 64       |
| eps_start       | Epsilon start value for exploration/exploitation   | 1.0      |
| eps_decay       | Epsilon decay value for exploration/exploitation   | 0.995    |
| eps_end         | Epsilon minimum value for exploration/exploitation | 0.01     |
| gamma           | The reward discount factor                         | 0.99     |
| tau             | For soft update of target parameters               | 1e-3     |
| lr              | The learning rate                                  | 5e-4     |

## Ideas for future work

- Evaluate other neural network architectures: Moving from a 3-layer MLP to a 4-layer MLP network impacted the learning time significantly. This evidence suggests that other models such as convolutional or recurrent networks may bring additional gains.
- Reward noise reduction: Looking at the reward plot above, it seems the reward has great variance over the 100 episodes. The papers referenced in this report raise different ideas about how to deal with this issue. 

## References

- [Human-Level Control through Deep Reinforcement Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)