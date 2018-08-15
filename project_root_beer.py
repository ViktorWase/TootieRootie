from random import random

def norm_square(x):
	return sum(a*a for a in x)

def newton_raphson(func, func_der, parameters, dims, max_iter=100, convg_lim=1.0e-8, x0=None):
	if x0==None:
		x = [1.0e-8 for _ in range(dims)]
	else:
		assert dims == len(x0)
		x = x0

	for i in range(max_iter):
		f = func(x, parameters)
		f_dx = func_der(x, parameters)
		x = [x[i] - f[i]/f_dx[i] for i in range(dims)]
		if norm_square(f)<convg_lim*convg_lim:
			break
	return x


def main_loop(func_str, dims, nr_of_parameters, parameter_ranges, nr_of_derivatives=1):

	# Step 0
	# Generate the parameter points in the range
	nr_of_samples_per_parameter = 10 
	assert nr_of_parameters == len(parameter_ranges)

	nr_of_samples = 1
	for _ in range(nr_of_parameters):
		nr_of_samples *= nr_of_samples_per_parameter

	# Generate random parameter points in the given range.(stupid, change later)
	parameter_samples = [[0.0 for _ in range(nr_of_parameters)] for _ in range(nr_of_samples)]
	for i in range(nr_of_samples):
		for d in range(nr_of_parameters):
			parameter_samples[i][d] = random()*(parameter_ranges[d][1]-parameter_ranges[d][0])+parameter_ranges[d][0]

	# Let's convert the function string into a proper function
	# TODO: NEVER CODE LIKE THIS AGAIN! YOU ARE A DISGRACE TO PYTHON!
	func = eval("lambda x, a:"+func_str)

	assert nr_of_derivatives==1 # TODO: Fix the other cases?
	assert dims == 1 # TODO: Yeah. Generalize!
	# Let's get the derivative as well
	func_der = lambda x, a: (func([tmp+1.0e-8 for tmp in x], a)- func(x,a))/1.0e-8 # TODO: Do this symbolically, not numerically.

	# Step 1 
	# For each point, find the x val that is the (or a) root.
	# Do this using Newton-Raphson.
	root_samples = [newton_raphson_1d(func, func_der, parameter_samples[i], dims) for i in range(nr_of_samples)] # TODO: Deal with the points that didn't converge.

	# Step 2
	# Run a symbolic regression to find a good approximation for the root.
	# This is used as a starting point.

	# Step 3
	# Run a symbolic regression algorithm that finds an approximation
	# for the function defined by func_str. However, note that the
	# approximation function has to:
	# 1) have an analytical root.
	# 2) have p parameters (where p is nr_of_derivatives+1). These parameters have
	#    to be calculated analytically from the function and its derivatives (in a fixed point).

	# (Step 4 - Do a local search. Do later)

if __name__ == '__main__':
	from math import pi, sin
	main_loop('x[0]-a[0]*sin(x[0])-a[1]', 1, 2, [[0.0, 1.0],[0.0, 2*pi]])