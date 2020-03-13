import argparse
from itertools import count

import os, sys, random
import numpy as np
import _pickle as pickle 

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from utils.models import QNetwork, GaussianPolicy
from utils.ReplayBuffer import ReplayBuffer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SAC():
	def __init__(self, args):
		self.args = args
		self.env = gym.make(self.args.env_name)
		state_dim = self.env.observation_space.shape[0]
		action_dim = self.env.action_space.shape[0]

		self.actor = GaussianPolicy(state_dim, action_dim, 64, self.env.action_space).to(device)
		self.actor_optimizer = optim.Adam(self.actor.parameters(), self.args.lr)

		self.critic_1 = QNetwork(state_dim, action_dim, 64).to(device)
		self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(),self.args.lr)
		self.critic_target_1 = QNetwork(state_dim, action_dim, 64).to(device)
		self.critic_target_1.load_state_dict(self.critic_1.state_dict())

		self.critic_2 = QNetwork(state_dim, action_dim, 64).to(device)
		self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), self.args.lr)
		self.critic_target_2 = QNetwork(state_dim, action_dim, 64).to(device)
		self.critic_target_2.load_state_dict(self.critic_2.state_dict())

		self.replay_buffer = ReplayBuffer(self.args.capacity)

		self.global_steps = 0
		self.writer = SummaryWriter("log/" + self.args.env_name)
		log_file = "log/" + self.args.env_name + "_SAC.pck"
		if os.path.exists(log_file):
			os.remove(log_file)
		self.log_file = open(log_file, 'ab')

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

			# get the next action and compute target Q
			with torch.no_grad():
				next_action, log_prob, _ = self.actor.sample(next_state)
				target_Q1 = self.critic_target_1(next_state, next_action)
				target_Q2 = self.critic_target_2(next_state, next_action)
				target_Q = torch.min(target_Q1, target_Q2) - self.args.alpha * log_prob
				y_Q = reward + self.args.gamma * (1 - done) * target_Q

			# update critic
			current_Q1 = self.critic_1(state, action)
			critic_loss1 = F.mse_loss(current_Q1, y_Q)
			self.critic_optimizer_1.zero_grad()
			critic_loss1.backward()
			self.critic_optimizer_1.step()

			current_Q2 = self.critic_2(state, action)
			critic_loss2 = F.mse_loss(current_Q2, y_Q)
			self.critic_optimizer_2.zero_grad()
			critic_loss2.backward()
			self.critic_optimizer_2.step()

			# update actor
			actor_action, actor_log_prob, _ = self.actor.sample(state)
			Q1 = self.critic_1(state, actor_action)
			Q2 = self.critic_2(state, actor_action)
			actor_loss = -(torch.min(Q1, Q2) - self.args.alpha * actor_log_prob).mean()
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# update target network
			for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
				target_param.data.copy_((1-self.args.tau) * target_param.data + self.args.tau * param.data)

			for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
				target_param.data.copy_((1-self.args.tau) * target_param.data + self.args.tau * param.data)

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
					ep_r = 0
					break

			if len(self.replay_buffer.storage) >= self.args.capacity - 1:
				self.update()

		self.save(i+1)

	def evaluate(self, number = 1, render = True):
		rewards = []
		for _ in range(number):
			state = self.env.reset()
			done = False
			total_rews = 0
			time_step = 0
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

				# if time_step > 1000:
				# 	print("time out")
				# 	break
			if render:
				print("total reward of this episode is " + str(total_rews))
			rewards.append(total_rews)
		rewards = np.array(rewards)
		if not render:
			pickle.dump((self.global_steps, rewards), self.log_file)
		return rewards.max(), rewards.min(), rewards.mean()

	def close(self):
		self.env.close()
		self.writer.close()
		self.log_file.close()

	def save(self, episode):
		if episode == None:
			file_name = "weights/" + self.args.env_name + "_SAC_checkpoint.pt"
		else:
			file_name = "weights/" + self.args.env_name + "_SAC_checkpoint_" + str(episode) + ".pt"
		torch.save({'actor' : self.actor.state_dict(),
					'critic_1' : self.critic_1.state_dict(),
					'critic_2' : self.critic_2.state_dict(),
					'critic_target_1' : self.critic_target_1.state_dict(),
					'critic_target_2' : self.critic_target_2.state_dict()}, file_name)
		print("save model to " + file_name)

	def load(self, episode):
		if episode == None:
			file_name = "weights/" + self.args.env_name + "_SAC_checkpoint.pt"
		else:
			file_name = "weights/" + self.args.env_name + "_SAC_checkpoint_" + str(episode) + ".pt"
		checkpoint = torch.load(file_name)
		self.actor.load_state_dict(checkpoint['actor'])
		self.critic_1.load_state_dict(checkpoint['critic_1'])
		self.critic_2.load_state_dict(checkpoint['critic_2'])
		self.critic_target_1.load_state_dict(checkpoint['critic_target_1'])
		self.critic_target_2.load_state_dict(checkpoint['critic_target_2'])
		print("successfully load model from " + file_name)
