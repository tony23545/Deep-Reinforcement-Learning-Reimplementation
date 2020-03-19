import argparse
import os
import numpy as np 
import _pickle as pickle 
import gym
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

from utils.models import ValueNetwork, GaussianFixstdPolicy
from algorithms import algorithms

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state', 'done'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PPO(algorithms):
	clip_param = 0.2
	max_grad_norm = 40
	ppo_epoch = 5
	buffer_capacity = 2048
	batch_size = 32

	def __init__(self, args):
		super().__init__(args)
		num_state = self.env.observation_space.shape[0]
		num_action = self.env.action_space.shape[0]

		self.actor = GaussianFixstdPolicy(num_state, num_action, 64, self.env.action_space)
		self.actor_optimizer = optim.Adam(self.actor.parameters(), self.args.lr)

		self.critic = ValueNetwork(num_state, 64)

		self.critic_optimizer = optim.Adam(self.critic.parameters(), self.args.lr)

		self.buffer = []
		self.counter = 0
		self.training_step = 0
		self.global_steps = 0
		
		if self.args.last_episode > 0:
			self.load(self.args.last_episode)

	def store_transiction(self, transition):
		self.buffer.append(transition)
		self.counter += 1
		return self.counter % self.buffer_capacity == 0

	def compute_returns(self, next_value, rewards, dones):
		R = next_value
		returns = []
		for step in reversed(range(len(rewards))):
			R = rewards[step] + self.args.gamma * R
			returns.insert(0, R)
		return torch.FloatTensor(returns)

	def update(self):
		self.training_step += 1
		
		state = torch.FloatTensor([t.state for t in self.buffer])
		action = torch.FloatTensor([t.action for t in self.buffer])
		rewards = torch.FloatTensor([t.reward for t in self.buffer]).view(-1, 1)
		next_state = torch.FloatTensor([t.next_state for t in self.buffer])
		old_action_log_prob = torch.FloatTensor([t.a_log_prob for t in self.buffer]).view(-1, 1)
		dones = torch.FloatTensor([t.done for t in self.buffer]).view(-1, 1)

		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
		with torch.no_grad():
			target_v = rewards + self.args.gamma * self.critic(next_state)
			advantage = (target_v - self.critic(state))

		for _ in range(self.ppo_epoch):
			for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
				action_log_prob, entropy = self.actor.action_log_prob(state[index], action[index])
				ratio = torch.exp(action_log_prob - old_action_log_prob[index])

				L1 = ratio * advantage[index]
				L2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage[index]
				
				action_loss = -torch.min(L1, L2).mean() - 0.02 * entropy.sum(-1).mean()
				self.actor_optimizer.zero_grad()
				action_loss.backward()
				#print(action_loss)
				nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
				self.actor_optimizer.step()

				value_loss = F.smooth_l1_loss(self.critic(state[index]), target_v[index])
				self.critic_optimizer.zero_grad()
				value_loss.backward()
				nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
				self.critic_optimizer.step()

		del self.buffer[:]

	def train(self):
		self.actor.train()
		for i_epoch in range(self.args.max_episode):
			score = 0
			state = self.env.reset()
			for t in range(self.args.max_length_trajectory):
				action, action_log_prob, _ = self.actor.sample(torch.FloatTensor([state]).to(device))
				action = action.cpu().detach().numpy()[0]
				action_log_prob = action_log_prob.cpu().detach().numpy()[0]

				#self.env.render()
				next_state, reward, done, info = self.env.step(action)
				trans = Transition(state, action, reward, action_log_prob, next_state, done)

				if self.store_transiction(trans):
					self.update()

				score += reward
				state = next_state

				if done:
					break

			if i_epoch % self.args.print_log == 0:
				print(self.actor.action_log_std.exp().detach())
				print("Ep_i \t {}, time step {}, global_steps is {}".format(i_epoch, t, self.counter))
				self.evaluate(10, False)

		self.save(i_epoch+1)

	def evaluate(self, number = 1, render = True):
		self.actor.eval()
		rewards = []
		for _ in range(number):
			done = False
			total_rews = 0
			count = 0
			state = self.env.reset()
			while not done:
				with torch.no_grad():
					action, _, _ = self.actor.sample(torch.FloatTensor([state]).to(device))
					action = action.cpu().detach().numpy()[0]
				if render:
					self.env.render()
				state, reward, done, _ = self.env.step(action)

				total_rews += reward
				count += 1

			rewards.append(total_rews)
			if render:
				print("total reward of this episode is " + str(total_rews))
		rewards = np.array(rewards)
		if not render:
			pickle.dump((self.counter, rewards), self.log_file)
		self.actor.train()
		print("mean reward {}, max reward {}".format(rewards.mean(), rewards.max()))

	def save(self, episode = None):
		file_name = self.weights_file(episode)
		torch.save({'actor' : self.actor.state_dict(),
					'critic' : self.critic.state_dict()}, file_name)
		print("save model to " + file_name)

	def load(self, episode = None):
		file_name = self.weights_file(episode)
		checkpoint = torch.load(file_name)
		self.actor.load_state_dict(checkpoint['actor'])
		self.critic.load_state_dict(checkpoint['critic'])
		print("successfully load model from " + file_name)
