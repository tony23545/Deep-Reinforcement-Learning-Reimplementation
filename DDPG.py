import argparse
from itertools import count

import os, sys, random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from utils.models import QNetwork, DeterministicPolicy
from utils.ReplayBuffer import ReplayBuffer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DDPG(object):
	def __init__(self, args):
		self.args = args
		self.env = gym.make(self.args.env_name)
		state_dim = self.env.observation_space.shape[0]
		action_dim = self.env.action_space.shape[0]

		self.actor = DeterministicPolicy(state_dim, action_dim, 64, self.env.action_space).to(device)
		self.actor_target = DeterministicPolicy(state_dim, action_dim, 64, self.env.action_space).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = optim.Adam(self.actor.parameters(), self.args.lr)

		self.critic = QNetwork(state_dim, action_dim, 64).to(device)
		self.critic_target = QNetwork(state_dim, action_dim, 64).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = optim.Adam(self.critic.parameters(), self.args.lr)

		self.replay_buffer = ReplayBuffer(self.args.capacity)
		self.num_critic_update_iteration = 0
		self.num_actor_update_iteration = 0
		self.num_training = 0
		self.global_steps = 0
		self.writer = SummaryWriter("log/" + self.args.env_name)

		if self.args.last_episode > 0:
			self.load(self.args.last_episode)

	def update(self):
		for it in range(self.args.update_iteration):
			# sample from replay buffer
			x, y, u, r, d = self.replay_buffer.sample(self.args.batch_size)
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(u).to(device)
			next_state = torch.FloatTensor(y).to(device)
			done = torch.FloatTensor(d).to(device)
			reward = torch.FloatTensor(r).to(device)

			# computer the target Q value
			next_action, _, _ = self.actor_target.sample(next_state)
			target_Q = self.critic_target(next_state, next_action)
			target_Q = reward + ((1-done) * self.args.gamma * target_Q).detach()

			# get current Q estimate
			current_Q = self.critic(state, action)

			# compute cirtic loss and update
			critic_loss = F.mse_loss(current_Q, target_Q)
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# computer actor loss
			actor_action, _, _ = self.actor.sample(state)
			actor_loss = -self.critic(state, actor_action).mean()
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# update target model 
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

			self.num_actor_update_iteration += 1
			self.num_critic_update_iteration += 1

	def train(self):
		for i in range(self.args.max_episode):
			state = self.env.reset()
			ep_r = 0
			for t in count():
				state = torch.FloatTensor([state])
				action, _, _ = self.actor.sample(state)
				action = action.cpu().detach().numpy()[0]

				next_state, reward, done, info = self.env.step(action)
				self.global_steps += 1
				ep_r += reward
				self.replay_buffer.push((state, next_state, action, reward, np.float(done)))
				state = next_state

				if done or t > self.args.max_length_trajectory:
					if i % self.args.print_log == 0:
						print("Ep_i \t {}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
						self.evaluate(10, False)
					break

			if len(self.replay_buffer.storage) >= self.args.capacity - 1:
				self.update()
		self.save(i+1)


	def evaluate(self, number = 1, render = True):
		rewards = []
		for _ in range(number):
			total_rews = 0
			time_step = 0
			done = False
			state = self.env.reset()
			while not done:
				state = torch.FloatTensor([state])
				with torch.no_grad():
					# use the mean action
					_, _, action = self.actor.sample(state)
					action = action.cpu().detach().numpy()[0]
				if render:
					self.env.render()
				state, reward, done, _ = self.env.step(action)
				total_rews += reward
				time_step += 1

				if time_step > 1000:
					print("time out")
					break
			if render:
				print("total reward of this episode is " + str(total_rews))
			rewards.append(total_rews)
		rewards = np.array(rewards)
		if not render:
			self.writer.add_scalar('DDPG_reward',rewards.mean(), self.global_steps)
		return rewards.max(), rewards.min(), rewards.mean()

	def close(self):
		self.env.close()
		self.writer.close()

	def load(self, episode = None):
		if episode == None:
			file_name = "weights/" + self.args.env_name + "_DDPG_checkpoint.pt"
		else:
			file_name = "weights/" + self.args.env_name + "_DDPG_checkpoint_" + str(episode) + ".pt"
		checkpoint = torch.load(file_name)
		self.actor.load_state_dict(checkpoint['actor'])
		self.actor_target.load_state_dict(checkpoint['actor_target'])
		self.critic.load_state_dict(checkpoint['critic'])
		self.critic.load_state_dict(checkpoint['critic_target'])
		print("successfully load model from " + file_name)

	def save(self, episode = None):
		if episode == None:
			file_name = "weights/" + self.args.env_name + "_DDPG_checkpoint.pt"
		else:
			file_name = "weights/" + self.args.env_name + "_DDPG_checkpoint_" + str(episode) + ".pt"
		torch.save({'actor' : self.actor.state_dict(),
					'critic' : self.critic.state_dict(),
					'actor_target' : self.actor_target.state_dict(),
					'critic_target' : self.critic_target.state_dict()}, file_name)
		print("save model to " + file_name)
