import argparse

from SAC import SAC
from TD3 import TD3
from DDPG import DDPG
from PPO import PPO
from A2C import A2C


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default='Pendulum-v0')
parser.add_argument('--model', default='SAC')
parser.add_argument('--mode', default='train')
parser.add_argument('--num_envs', default=8)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--alpha', default=0.2, type=float)

parser.add_argument('--capacity', default=50000, type=int) # replay buffer size

parser.add_argument('--max_episode', default=2000, type=int) # num of games
parser.add_argument('--last_episode', default=0, type=int)
parser.add_argument('--max_length_trajectory', default=5000, type=int)
parser.add_argument('--print_log', default=50, type=int)
parser.add_argument('--exploration_noise', default=0.1)
parser.add_argument('--policy_delay', default=2)

parser.add_argument('--update_iteration', default=10, type=int)
parser.add_argument('--batch_size', default=64, type=int) # mini batch size

# experiment relater
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--exp_name', default='experiment')
args = parser.parse_args()

def main():
	if args.model == "TD3":
		agent = TD3(args)
	elif args.model == "DDPG":
		agent = DDPG(args)
	elif args.model == "PPO":
		agent = PPO(args)
	elif args.model == "A2C":
		agent = A2C(args)
	else:
		agent = SAC(args)

	if args.mode == 'train':
		agent.train()
	elif args.mode == 'eval':
		agent.evaluate(5)
	agent.close()

if __name__ == '__main__':
	main()