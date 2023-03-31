# Reinforcement Learning Methods

Reinforcement learning (RL) is an area of machine learning that deal with decision making processes. RL is defined by two basic components: Environment and Agent. Environment dictates the world and its rules. The agent is an entity that perceives the environment and can act based on what it saw. Each time the agent takes an action, the environment changes and it receives an reward based on that change. The goal of the agent is to maximize the total reward it receives over time.

Firstly created for personal use and learning purposes. This project now has the objective to provide simple implementation of reinforcement learning methods. It was created to be modular, so each component could be easly modified. Currently there are two methods and improvements: Q-Learning and Deep Q Networks. In the future, new methods and features will be included as well as a wiki containing explanation of the methods and a documentation.

The methods interfaces are based on the gymnasium environments.

# Prerequisites
- python3
- numpy
- gymnasium
- pytorch

# How to use

To create an agent, we first import the gymnasium. Them, import the Method from the DQN package and create it with the desired hiperparameters. Some values expect an Schedule. It can be a Constant value over all trainning or a Linear which changes the value it receives in a constant rate. 

After agent creation, just call the train function. It takes an environment and the number of steps. At the end, we save the agent data in the directory provided named as the prefix given. 

```rb
import gymnasium as gym

from RL_Methods.DQN import DQNAgent
from RL_Methods.utils.Schedule import ConstantSchedule, LinearSchedule

env = gym.make("CartPole-v1")
dqnAgent = DQNAgent(
    env.observation_space.shape,
    env.action_space.n,
    learning_rate=ConstantSchedule(0.0001),
    epsilon=LinearSchedule(1.0, -1e-5, 0.05),
    gamma=0.9,
    batch_size=64,
    experience_buffer_size=1e5,
    target_network_sync_freq=2000,
    grad_norm_clip=1,
    )
dqnAgent.train(env, 500000)
dqnAgent.save("data/", prefix="dqn")
```


# TODO

- [] Convolutional Policy
- [] Distributed

- [x] Q-Learning: https://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf
    - [] Observation space discretization
    - [x] Double Q-Learning: https://papers.nips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf
    - [] Aproximated Q-Learning

- [x] DQN: 
    - [x] Double: https://arxiv.org/abs/1509.06461
    - [x] Dueling: https://arxiv.org/abs/1511.06581
    - [x] Multistep: https://arxiv.org/abs/1901.07510
    - [x] Distributional: https://arxiv.org/abs/1707.06887
    - [x] Noisy: https://arxiv.org/abs/1706.10295
    - [x] Prioritized: https://arxiv.org/abs/1511.05952v4
    - [] Boltzmann Softmax: 
    - [x] Rainbow: https://arxiv.org/abs/1710.02298

- [] REINFORCE
- [] PPO
- [] DDPG
- [] A2C
- [] A3C
- [] TRPO
