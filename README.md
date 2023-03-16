# Reinforcement Learning Methods

Reinforcement learning (RL) is an area of machine learning that deal with decision making processes. RL is defined by two basic components: Environment and Agent. Environment dictates the world and its rules. The agent is an entity that perceives the environment and can act based on what it saw. Each time the agent takes an action, the environment changes and it receives an reward based on that change. The goal of the agent is to maximize the total reward it receives over time.

In this project there implementation of Reinforcement Learning methods. Currently there are two methods and some improvements to them: Q-Learning and Deep Q Networks. 

# Prerequisites
- numpy
- gymnasium
- pytorch


# TODO

- [] Convolutional Policy
- [] Distributed

- [x] Q-Learning
    - [] Observation space discretization
    - [x] Double Q-Learning
    - [] Aproximated Q-Learning

- [x] DQN
    - [x] Double
    - [x] Dueling
    - [x] Multistep
    - [x] Distributional
    - [x] Noisy
    - [x] Prioritized
    - [] Boltzmann Softmax
    - [x] Rainbow

- [] REINFORCE
- [] PPO
- [] DDPG
- [] A2C
- [] A3C
- [] TRPO