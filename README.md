# Deep Reinforcement Learning Reimplementation
This is my final project for [cse573: Artificial Intelligence](https://courses.cs.washington.edu/courses/cse573/20wi/). In this project, I reimplement 5 state-of-the-art algorithms (A2C, DDPG, PPO, TD3 and SAC) and carry out some experiments to study the effects of different aspects on the performance of models. This repo only serves for learning purpose and still has many difference from the published baseline. I borrow some ideas from [sweetice](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch)'s repo during implementation.

## Basic Usage
For example, to train TD3 on Hopper-v2 environment for 2000 episode, simply use

```
python main.py --model TD3 --env_name Hopper-v2 --max_episode 2000
```

To evaluate the training result

```
python main.py --model TD3 --env_name Hopper-v2 --last_episode 2000 --mode eval
```
There are also many other options sepcified in the `main.py` file. For example, change the random seed to 10 and the capacity of replay buffer to 10000
```
python main.py --model TD3 --env_name Hopper-v2 --max_episode 2000 --seed 10 --capacity 10000
```

To visualize the training log

```
python plot_result.py --dir log/Hopper-v2/TD3
```
