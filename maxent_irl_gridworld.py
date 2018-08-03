
# import modules
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

# import me-irl and utilities
import img_utils
from maxent_irl import *
from mdp import gridworld
from mdp import value_iteration

np.random.seed(1)

def feature_coord(gw):
	"""Generates feature matrix for grid world"""
	N = gw.height * gw.width
	feat = np.zeros([N, 2])
	
	for i in range(N):
		iy, ix = gw.idx2pos(i)
		feat[i,0] = iy
		feat[i,1] = ix
	
	return feat

def feature_basis(gw):
	"""
	Generates a NxN feature map for gridworld
	input:
		gw		Gridworld
	returns
		feat	NxN feature map - feat[i, j] is the l1 distance between state i and state j
	"""
	N = gw.height * gw.width
	feat = np.zeros([N, N])
	for i in range(N):
		for y in range(gw.height):
			for x in range(gw.width):
				iy, ix = gw.idx2pos(i)
				feat[i, gw.pos2idx([y, x])] = abs(iy-y) + abs(ix-x)
	
	return feat

def generate_demonstrations(gw, policy, n_trajs=100, len_traj=20, rand_start=False, start_pos=[0,0]):
	"""Gathers expert demonstrations
	inputs:
		gw          Gridworld - the environment
		policy      Nx1 matrix
		n_trajs     int - number of trajectories to generate
		rand_start  bool - randomly picking start position or not
		start_pos   2x1 list - set start position, default [0,0]
	returns:
		trajs       a list of trajectories - each element in the list is a list of Steps representing an episode
	"""

	trajs = []
	for i in range(n_trajs):
		if rand_start:
			# override start_pos
			start_pos = [np.random.randint(0, gw.height), np.random.randint(0, gw.width)]

		episode = []
		gw.reset(start_pos)
		cur_state = start_pos
		cur_state, action, next_state, reward, is_done = gw.step(int(policy[gw.pos2idx(cur_state)]))
		episode.append(Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward, done=is_done))

		for _ in range(len_traj):
			cur_state, action, next_state, reward, is_done = gw.step(int(policy[gw.pos2idx(cur_state)]))
			episode.append(Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward, done=is_done))
			if is_done:
				break
		trajs.append(episode)
	
	return trajs

	
def generate_random(gw, n_actions, n_trajs=100, len_traj=20, rand_start=False, start_pos=[0,0]):
	"""gatheres random demonstrations

	inputs:
	gw          Gridworld - the environment
	policy      Nx1 matrix
	n_trajs     int - number of trajectories to generate
	rand_start  bool - randomly picking start position or not
	start_pos   2x1 list - set start position, default [0,0]
	returns:
	trajs       a list of trajectories - each element in the list is a list of Steps representing an episode
	"""

	trajs = []
	for i in range(n_trajs):
		if rand_start:
			# override start_pos
			start_pos = [np.random.randint(0, gw.height), np.random.randint(0, gw.width)]

		episode = []
		gw.reset(start_pos)
		cur_state = start_pos
		cur_state, action, next_state, reward, is_done = gw.step(np.random.randint(n_actions))
		episode.append(Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward, done=is_done))
		
		# while not is_done:
		for _ in range(len_traj):
			cur_state, action, next_state, reward, is_done = gw.step(np.random.randint(n_actions))
			episode.append(Step(cur_state=gw.pos2idx(cur_state), action=action, next_state=gw.pos2idx(next_state), reward=reward, done=is_done))
			if is_done:
				break
		
		trajs.append(episode)

	return trajs

def main():

	# named tuple to record demonstrations
	Step = namedtuple('Step','cur_state action next_state reward done')

	# argument parser for command line arguments
	parser = argparse.ArgumentParser(description=None)

	parser.add_argument('-wid', '--width', default=5, type=int, 
						help='width of the gridworld')
	parser.add_argument('-hei', '--height', default=5, type=int, 
						help='height of the gridworld')
	parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, 
						help='learning rate')
	parser.add_argument('-l', '--l_traj', default=20, type=int, 
						help='length of expert trajectory')

	parser.add_argument('--no-rand_start', dest='rand_start', action='store_false', 
						help='when sampling trajectories, fix start positions')
	parser.add_argument('--rand_start', dest='rand_start', action='store_true', 
						help='when sampling trajectories, randomly pick start positions')
	parser.add_argument('--approx', dest='approx', action='store_true', 
						help='flag to perform approximation of psa')

	parser.add_argument('-g', '--gamma', default=0.9, type=float, 
						help='discount factor')
	parser.add_argument('-n', '--n_iters', default=20, type=int, 
						help='number of iterations')
	parser.add_argument('-t', '--n_trajs', default=100, type=int, 
						help='number of expert trajectories')
	parser.add_argument('-a', '--act_random', default=0.3, type=float, 
						help='probability of acting randomly')
	
	# set default value for rand_start variable
	parser.set_defaults(rand_start=False)

	# parse and print arguments
	args = parser.parse_args()

	# arguments for environment and irl algorithm
	r_max = 1 
	gamma = args.gamma
	width = args.width
	height = args.height
	l_traj = args.l_traj
	approx = args.approx
	n_iters = args.n_iters
	n_trajs = args.n_trajs
	act_rand = args.act_random
	rand_start = args.rand_start
	learning_rate = args.learning_rate

	# variables for number of actions and states
	n_actions = 5
	n_states = height * width

	# initialize the gridworld
	# rmap_gt is the ground truth for rewards
	rmap_gt = np.zeros([height, width])

	rmap_gt[0, width-1] = r_max
	rmap_gt[height-1, 0] = r_max
	rmap_gt[height-1, width-1] = r_max

	# create grid world instance
	gw = gridworld.GridWorld(rmap_gt, {}, 1-act_rand)

	# get true rewards, state transition dynamics
	rewards_gt = np.reshape(rmap_gt, height*width, order='F')
	P_a_true = gw.get_transition_mat()

	trajs = generate_random(gw, n_actions, n_trajs=n_trajs, len_traj=l_traj, rand_start=rand_start)

	# get approximation of state transition dynamics
	P_a_approx = np.zeros((n_states, n_states, n_actions))
	for traj in trajs:
		for t in range(len(traj)):
			P_a_approx[traj[t].cur_state, traj[t].next_state, traj[t].action] += 1

	for s in range(n_states):
		for a in range(n_actions):
			if np.sum(P_a_approx[s,:,a]) != 0:
				P_a_approx[s,:,a] /= np.sum(P_a_approx[s,:,a])

	if approx:
		P_a = P_a_approx
	else:
		P_a = P_a_true

	# get true value function and policy from reward map
	values_gt, policy_gt = value_iteration.value_iteration(P_a, rewards_gt, gamma, error=0.01, deterministic=True)

	# use identity matrix as feature
	feat_map = np.eye(n_states)

	# other two features. due to the linear nature, 
	# the following two features might not work as well as the identity.
	# feat_map = feature_basis(gw)
	# feat_map = feature_coord(gw)

	trajs = generate_demonstrations(gw, policy_gt, n_trajs=n_trajs, len_traj=l_traj, 
									rand_start=rand_start)

	# perform inverse reinforcement learning to get reward function
	rewards = maxent_irl(feat_map, P_a, gamma, trajs, learning_rate, n_iters)
	values, _ = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)

	# plots
	plt.figure(figsize=(20,4))
	plt.subplot(2, 2, 1)
	img_utils.heatmap2d(rmap_gt, 'Rewards Map - Ground Truth', block=False)
	plt.subplot(2, 2, 2)
	img_utils.heatmap2d(np.reshape(values_gt, (height,width), order='F'), 'Value Map - Ground Truth', block=False)
	plt.subplot(2, 2, 3)
	img_utils.heatmap2d(np.reshape(rewards, (height,width), order='F'), 'Reward Map - Recovered', block=False)
	plt.subplot(2, 2, 4)
	img_utils.heatmap2d(np.reshape(values, (height,width), order='F'), 'Value Map - Recovered', block=False)
	plt.show()

	# plots for state transition dynamics
	plt.figure(figsize=(10,4))
	plt.subplot(2, 1, 1)
	img_utils.heatmap2d(np.reshape(P_a_true[10,:,2], (height,width), order='F'), 'True Dist', block=False)
	plt.subplot(2, 1, 2)
	img_utils.heatmap2d(np.reshape(P_a_approx[10,:,2], (height,width), order='F'), 'Approx Dist', block=False)
	plt.show()

if __name__ == "__main__":
  main()