import numpy as np
from numba import njit, bool_


@njit
def choice(p):
	"""Because np.random.choice is horribly slow we write our own.
	Fastest implementation I could think of to do a fast weighted bool choice"""
	if np.random.random() > p:
		return False
	else:
		return True


# We only call this once, makes no sense to jit it
#@njit
def Nchoice(p, N=1):
	"""Because np.random.choice is horribly slow we write our own.
	Fastest implementation I could think of to do a fast weighted bool choice"""
	out = np.ones(N, dtype=np.bool_)

	# Loops are OK if we use numba- use this if you jit it
	#for i in range(N):
	#	if np.random.random() > p:
	#		out[i] = False

	# If you don't jit:
	x = np.random.random(N)
	out[np.argwhere(x > p)] = False
	return out