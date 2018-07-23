"""Utility functions for inverse reinforcement learning"""
import math
import numpy as np
from collections import namedtuple

# named tuple for processing
Step = namedtuple('Step','cur_state action next_state reward done')

# function to normalize input array
def normalize(vals):
	"""
	normalize to (0, max_val)
	input:
	vals: 1d array
	"""
	min_val = np.min(vals)
	max_val = np.max(vals)
	return (vals - min_val) / (max_val - min_val)

# function to estimate sigmoid function on input array
def sigmoid(xs):
	"""
	sigmoid function
	inputs:
		xs      1d array
	"""
	return [1 / (1 + math.exp(-x)) for x in xs]