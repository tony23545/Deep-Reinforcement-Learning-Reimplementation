import gym
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.distributions import Normal
from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter

from utils.models import ValueNetwork, GaussianPolicy
from utils.multiprocessing_env import SubprocVecEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env(env_name):
	def _thunk():
		env = gym.make(env_name)
		return env
	return _thunk

class A2C():
	def __init__(self, args):
		self.args = args
		envs = [make_env(self.args.env_name) for i in range(self.args.num_envs)]
		self.envs = SubprocVecEnv(envs)
		state_dim = self.envs.observation_space.shape[0]
		action_dim = self.envs.action_space.shape[0]
		self.eps = np.linspace(0, 0.5, self.args.num_envs)

		self.actor = GaussianPolicy(state_dim, action_dim, 64, self.envs.action_space)
		self.actor_optimizer = optim.Adam(self.actor.parameters(), self.args.lr)
		self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.actor_optimizer, gamma=0.9)

		self.critic = ValueNetwork(state_dim, 64)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), self.args.lr)
		self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.critic_optimizer, gamma=0.9)
		self.global_steps = 0
		self.writer = SummaryWriter("log/" + self.args.env_name)

		if self.args.last_episode > 0:
			try:
				self.load(self.args.last_episode)
			except:
				print("can't find last checkpoint file")

		# set reandom seed
		self.env.seed(self.args.seed)
		torch.manual_seed(args.seed)
		np.random.seed(self.args.seed)
		

	def compute_returns(self, next_value, rewards, dones):
		R = next_value
		returns = []
		for step in reversed(range(len(rewards))):
			R = rewards[step] + self.args.gamma * R
			returns.insert(0, R)
		return returns

	def get_value(self, state):
		state = torch.FloatTensor(state)
		with torch.no_grad():
			value = self.critic(state)
		return value

	def evaluate(self, number = 1, render = True):
		env = gym.make(self.args.env_name)
		self.actor.eval()
		rewards = []
		for _ in range(number):
			state = env.reset()
			done = False
			total_rews = 0
			count = 0
			while not done:
				state = torch.FloatTensor([state]).to(device)
				with torch.no_grad():
					_, _, action = self.actor.sample(state)
				if render:
					env.render()
				state, reward, done, _ = env.step(action.cpu().numpy()[0])
				total_rews += reward
				count += 1
				if count > 1000:
					print("time out")
					break
			if render:
				print("total reward of this episode is " + str(total_rews))
			rewards.append(total_rews)
		env.close()
		rewards = np.array(rewards)
		if not render:
			self.writer.add_scalar('A2C_reward',rewards.mean(), self.global_steps)
		return rewards.max(), rewards.min(), rewards.mean()

	def train(self):
		state = self.envs.reset()
		episode_idx = self.args.last_episode

		self.actor.train()
		self.critic.train()

		while episode_idx < self.args.max_episode:
			log_probs = []
			states = []
			rewards = []
			dones = []

			# correct data
			for _ in range(self.args.max_length_trajectory):
				
				state_t = torch.FloatTensor(state).to(device)
				action, log_prob, _ = self.actor.sample(state_t, entropy = False)

				if True:
					random_action = torch.FloatTensor([self.envs.action_space.sample() for _ in range(self.args.num_envs)])
					explore = (np.random.random(self.args.num_envs) < self.eps)
					action[explore] = random_action[explore]

				next_state, reward, done, _ = self.envs.step(action.cpu().detach().numpy())
				self.global_steps += self.args.num_envs

				#value = self.get_value(state)
				log_probs.append(log_prob)
				states.append(state)
				rewards.append(reward)
				dones.append(done)

				state = next_state

			next_value = self.get_value(next_state).view(1, -1).cpu().numpy()
			returns = self.compute_returns(next_value, rewards, dones)

			log_probs = torch.cat(log_probs).view(-1, self.args.num_envs)
			returns = torch.FloatTensor(returns).view(-1, self.args.num_envs)
			states = torch.FloatTensor(states)
			values = self.critic(states).view(-1, self.args.num_envs)

			# update actor
			advantage = returns - values.detach()
			self.actor_optimizer.zero_grad()
			actor_loss = -(log_probs * advantage).sum() / self.args.num_envs
			actor_loss.backward()
			self.actor_optimizer.step()

			# update critic
			#values = self.critic(states).view(-1, args.num_envs)
			for _ in range(1):
				values = self.critic(states).view(-1, self.args.num_envs)
				self.critic_optimizer.zero_grad()
				critic_loss = F.smooth_l1_loss(values, returns)
				critic_loss.backward()
				self.critic_optimizer.step()

			episode_idx += 1

			if episode_idx % 200 == 0:
				self.actor_scheduler.step()
				self.critic_scheduler.step()
				self.eps = self.eps * 0.9
				
			
			if episode_idx % self.args.print_log == 0:
				print("epi {} best reward: {}".format(episode_idx, np.sum(rewards, axis = 0).max()))
				self.evaluate(10, False)
		self.save(episode_idx)

	def close(self):
		self.envs.close()
		self.writer.close()

	def save(self, episode = None):
		if episode == None:
			file_name = "weights/" + self.args.env_name + "_A2C_checkpoint.pt"
		else:
			file_name = "weights/" + self.args.env_name + "_A2C_checkpoint_" + str(episode) + ".pt"
		torch.save({'actor' : self.actor.state_dict(),
					'critic' : self.critic.state_dict()}, file_name)
		print("save model to " + file_name)


	def load(self, episode = None):
		if episode == None:
			file_name = "weights/" + self.args.env_name + "_A2C_checkpoint.pt"
		else:
			file_name = "weights/" + self.args.env_name + "_A2C_checkpoint_" + str(episode) + ".pt"
		checkpoint = torch.load(file_name)
		self.actor.load_state_dict(checkpoint['actor'])
		self.critic.load_state_dict(checkpoint['critic'])
		print("successfully load model from " + file_name)
