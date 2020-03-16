import torch
import os
import gym
import numpy as np

class algorithms():
	def __init__(self, args):
		self.args = args
		self.env = gym.make(self.args.env_name)

		log_file = "log/" + self.args.env_name + "/" + self.args.model + "/"
		if not os.path.exists(log_file):
			os.mkdir(log_file)
		log_file = log_file + self.args.exp_name + ".pck"
		if self.args.mode == 'train' and os.path.exists(log_file):
			os.remove(log_file)
		self.log_file = open(log_file, 'ab')

		# set reandom seed
		self.env.seed(self.args.seed)
		torch.manual_seed(self.args.seed)
		np.random.seed(self.args.seed)

	def weights_file(self, episode = None):
		file_name = "weights/" + self.args.env_name + "/" + self.args.model + "/"
		if not os.path.exists(file_name):
			os.mkdir(file_name)
		if episode == None:
			file_name = file_name + self.args.exp_name + ".pt"
		else:
			file_name = file_name + self.args.exp_name + "_" + str(episode) + ".pt"
		return file_name

	def close(self):
		self.env.close()
		self.log_file.close()