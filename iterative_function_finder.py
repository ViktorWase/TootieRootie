from random import random
from math import sqrt, fabs

from sys import version_info
if version_info >= (3,0):
	from math import inf
else:
	inf = 1.0e25

from types import *

from cgp import Operation
from simulated_anneling import multistart_opt
from non_linear_curve_fitting import combo_curve_fitting
from newton_raphson import root_finders
from nealder_mead import nealder_mead

def numerical_derivative(func, x, beta, der_number=1, h=1.0e-10):

	# NOTE: func has 1 input and 1 output, but as many parameters as needed.
	assert der_number>=1

	if der_number==1:
		return  (func([x[0]+h], beta)-func([x[0]-h], beta)) / (2*h)
	elif der_number==2:
		return (func([x[0]+h], beta)- 2*func(x, beta) +func([x[0]-h], beta)) / (h*h)
	elif der_number==3:
		return (-0.5*func([x[0]-2*h], beta)+func([x[0]-h], beta)-func([x[0]+h], beta)+0.5*func([x[0]+2*h], beta)) / (h*h*h)
	assert False

def numerical_derivative_dual(func, x, beta, der_number=1, h=1.0e-10):

	# NOTE: func has 1 input and 1 output, but as many parameters as needed.
	assert der_number>=1

	if der_number==1:
		f_x = func([x[0]+h], beta)
		f_mx = func([x[0]-h], beta)
		return ( (f_x[0]-f_mx[0]) / (2*h),  (f_x[1]-f_mx[1]) / (2*h))
	elif der_number==2:
		assert False
		return (func([x[0]+h], beta)- 2*func(x, beta) +func([x[0]-h], beta)) / (h*h)
	elif der_number==3:
		assert False
		return (-0.5*func([x[0]-2*h], beta)+func([x[0]-h], beta)-func([x[0]+h], beta)+0.5*func([x[0]+2*h], beta)) / (h*h*h)
	assert False

def residual_func(f_vals, pnts, beta, func, nr_of_derivatives):
	"""
	This function returns a vector of the differences between
	the actual function values (f_vals) and the value of the 
	function that we are trying to tune ( func(pnt, beta) ).

	Each element in the output vector represents one point from
	the input vector pnts.
	"""
	nr_of_pnts = len(pnts)
	differences = [None] * nr_of_pnts * (nr_of_derivatives+1)
	counter = 0
	for der_nr in range(nr_of_derivatives+1): 
		for i in range(nr_of_pnts):
			if der_nr == 0:
				tmp = func(pnts[i], parameters=beta) - f_vals[counter]
			else:
				tmp = numerical_derivative(func, pnts[i], beta, der_number=nr_of_derivatives)
			differences[counter] = tmp
			counter += 1
	return differences

def jacobian_func(f_vals, pnts, beta, func, nr_of_derivatives):
	def local_jacobi_help_func(X, parameters, derivative, der_dir, variable_derivation_number):
		if variable_derivation_number == 0:
			return func(X, parameters=beta, derivative=True, der_dir=der_dir)
		else:
			curry_func = lambda X, beta: func(X, parameters=beta, derivative=True, der_dir=der_dir)
			return numerical_derivative_dual(curry_func, X, beta, der_number=variable_derivation_number)

	nr_of_pars = len(beta)
	assert nr_of_pars > 0
	assert nr_of_derivatives >= 0

	nr_of_residuals = (1+nr_of_derivatives) * len(pnts)

	jacobian = [[None for i in range(nr_of_pars)] for j in range(nr_of_residuals)]

	# TODO: Will this crash if nr_of_derivatives is zero?
	for j in range(nr_of_pars):
		derivatives_counter = 0
		pnt_counter = 0
		for k in range(nr_of_residuals):
			if k%len(pnts) == 0 and k!=0:
				derivatives_counter += 1

				assert derivatives_counter <= nr_of_derivatives
			pnt = pnts[pnt_counter]
			(val, der) = local_jacobi_help_func(pnt, beta, True, len(pnt)+j, derivatives_counter)
			jacobian[k][j] = val*der

			pnt_counter += 1
			if pnt_counter == len(pnts):
				pnt_counter = 0
	# TODO: Rewrite this when the symbolic derivation is done.

	return jacobian

def get_parameter_pnts(nr_of_parameters_in_inp, parameter_ranges, variable_range, nr_of_samples_per_parameter, nr_of_variable_samples, should_repeat_pnts=False):
	# Calculate the total number of samples.
	nr_of_par_samples = 1
	for _ in range(nr_of_parameters_in_inp):
		nr_of_par_samples *= nr_of_samples_per_parameter

	# Generate random parameter points in the given range. 
	# TODO: this is stupid, change later! We really should sample on a nice
	# Cartesian grid.
	parameter_samples = [[0.0 for _ in range(nr_of_parameters_in_inp)] for _ in range(nr_of_par_samples)]
	for i in range(nr_of_par_samples):
		for d in range(nr_of_parameters_in_inp):
			parameter_samples[i][d] = random()*(parameter_ranges[d][1]-parameter_ranges[d][0])+parameter_ranges[d][0]

	assert variable_range[1] > variable_range[0]
	assert len(variable_range) == 2

	if should_repeat_pnts:
		pnts = [[[i*float(variable_range[1]-variable_range[0])/(nr_of_variable_samples-1)+variable_range[0]] for i in range(nr_of_variable_samples)] for _ in range(nr_of_par_samples)]
	else:
		pnts = [[i*float(variable_range[1]-variable_range[0])/(nr_of_variable_samples-1)+variable_range[0]] for i in range(nr_of_variable_samples)]

	return (pnts, parameter_samples, nr_of_par_samples)

def convergence_error(first_approx, input_function_and_derivatives, true_root_vals, parameter_in_inp_samples, cgp, nr_of_derivatives, nr_of_parameters_in_inp, cgp_parameters, max_root_iter=20, thresh=1.0e-8):
	"""
	The quality of the root solver can be measured by cheecking how often and fast it converges.

	In this case we are given a threshold to converge to and a maximum number of iterations to do it in. If
	the algorithm has manages to converge, then the number of iterations is added to the error. If it doesn't
	then the maximum iteration number is added, as well as a penalty and the magnitude of the last function-value.
	"""
	non_convergance_penalty = max_root_iter*0.5

	total_error = 0.0
	for i in range(len(parameter_in_inp_samples)):
		par = parameter_in_inp_samples[i]
		error = 0.0
		assert len(par) == nr_of_parameters_in_inp

		x = first_approx(par)
		has_converged = False
		f_and_d = None

		# Start the iterative improvement
		for itr in range(max_root_iter):
			try:
				f_and_d = [input_function_and_derivatives[k]([x], par) for k in range(nr_of_derivatives+1)]
			except (ValueError, ZeroDivisionError): # Sometimes x gets really big and this can cause problems.
				print("Math domain error in convergance_error.")
				error += 1.0e10
				break

			if fabs(f_and_d[0]) < thresh:
				has_converged = True
				last_iter = itr
				break

			# Create a data point for the cgp
			pnt = [x] + f_and_d + par

			try:
				# Calculate the new root approximation value
				x = cgp.eval(pnt, parameters=cgp_parameters)
			except ValueError: # Sometimes x gets really big and this can cause problems.
				print("Math domain error in CGP, convergance_error.")
				break

			# At 100000000000000000000 I say that it has diverged.
			# This is done to avoid math domain errors.
			if fabs(x) > 1.0e20:
				break

		# Calculate the error
		if not has_converged:
			error += max_root_iter
			error += non_convergance_penalty
			error += f_and_d[0]*f_and_d[0]
		else:
			error += last_iter
		total_error += error
	return total_error / float(len(parameter_in_inp_samples)) / (max_root_iter+non_convergance_penalty)

def direct_error_func(first_approx, input_function_and_derivatives, true_root_vals, parameter_in_inp_samples, dims, cgp, nr_of_cgp_parameters, op_table, nr_of_parameters_in_inp, nr_of_derivatives):
	"""
	This direct error function takes a different approach than the other error function. Instead
	of trying to approximate the function and then solving the system of equations for the next 
	root value, it directly develops a function that predicts the root value.
	"""

	used_vars_and_pars = cgp.which_variables_and_parameters_are_used()
	is_the_func_or_a_der_used = False
	for val in used_vars_and_pars:
		if val >= 1 and val <= nr_of_derivatives+1:
			is_the_func_or_a_der_used = True
			break
	if not is_the_func_or_a_der_used:
		return (1.0e23, [0.0]*cgp.nr_of_parameters)

	if cgp.is_constant:
		return (1.0e20, [])

	if cgp.nr_of_parameters == 0:
		are_pars_used = False
	else:
		are_pars_used = sum(cgp.which_parameters_are_used())==0

	if not are_pars_used:
		pars = [0.0]*cgp.nr_of_parameters
		return (convergence_error(first_approx, input_function_and_derivatives, true_root_vals, parameter_in_inp_samples, cgp, nr_of_derivatives, nr_of_parameters_in_inp, pars), pars)
		print("not numerical")
	else:
		p = cgp.nr_of_parameters

		def objective_func(pars):
			assert p == len(pars)
			return convergence_error(first_approx, input_function_and_derivatives, true_root_vals, parameter_in_inp_samples, cgp, nr_of_derivatives, nr_of_parameters_in_inp, pars)
		return nealder_mead(objective_func, p)

		#objective_func = lambda true_root_vals, parameter_in_inp_samples, beta, func: convergence_error(first_approx, input_function_and_derivatives, true_root_vals, parameter_in_inp_samples, cgp, nr_of_derivatives, nr_of_parameters_in_inp, beta)
		#(pars, err) = differential_evolution(parameter_in_inp_samples, true_root_vals, nr_of_cgp_parameters, objective_func, func, max_iter=50, pop_size=15)
		#return (err, pars)

def calc_function_and_derivatives(func, x_pnt, parameter_in_inp_sample, nr_of_derivatives):

	out = [func(x_pnt, parameter_in_inp_sample)]

	for i in range(1,nr_of_derivatives+1):
		out.append(numerical_derivative(func, x_pnt, parameter_in_inp_sample, der_number=i))
	return out

def direct_iterative_function_finder(start_func, input_function_and_derivatives, nr_of_parameters_in_inp, parameter_ranges, variable_range, optimizer, nr_of_derivatives=1, max_iter=1000, multi_starts=25, nr_of_samples_per_inp_parameter=8, nr_of_variable_samples=15, nr_of_cgp_pars=2, max_time=None):
	"""
	A root finder can be split into two parts: a function for x_0, and a function for x_{i+1} given x_i. This 
	method uses symbolic regression (read optimization) to find the latter.

	The procedure is simple really. The optimizer guesses at functions (in the form of Cartiesian Genetic Programs, CGPs)
	and then it tries to use that function as the iterative part of the root solving algorithm. If that works well it 
	gets a small error, if it works poorly it gets a big one. Simple.

	Note that the first part of the root finder (a function for x_0) is required, and is given as start_func. 
	"""

	# TODO: A lot of the code here is copied from the starting point approximation. Rewrite!

	assert nr_of_derivatives >= 0
	assert len(input_function_and_derivatives) == nr_of_derivatives+1
	func = input_function_and_derivatives[0]

	# The dimensions of the cgp are x, the value of the given function and its derivatives, as well
	# as the parameters of the input function.
	dims = 1 + nr_of_derivatives+1+nr_of_parameters_in_inp

	(x_pnts, parameter_in_inp_samples, nr_of_par_in_inp_samples) = get_parameter_pnts(nr_of_parameters_in_inp, parameter_ranges, variable_range, nr_of_samples_per_inp_parameter, nr_of_variable_samples)

	# Let's get the derivative as well. Symbolically if we have it, otherwise numerically.
	if len(input_function_and_derivatives)>=2:
		func_der = input_function_and_derivatives[1]
	else:
		func_der = lambda x, a: (func([x[0]+1.0e-11] , a) - func([x[0]],a))/1.0e-11 # TODO: Do this symbolically, not numerically.
	# Find the root values using Newton-Raphson. TODO: Add a bunch of root finders here to make sure that it converges.
	root_samples_and_errors = [root_finders(func, func_der, parameter_in_inp_samples[i]) for i in range(nr_of_par_in_inp_samples)]

	# Remove all points that didn't converge
	converge_thresh = 1.0e-8
	remove_idxs = []
	counter = 0
	for i in range(nr_of_par_in_inp_samples):
		err = root_samples_and_errors[counter][1]
		if err > converge_thresh:
			tmp = root_samples_and_errors.pop(counter)
			parameter_in_inp_samples.pop(counter)
			assert tmp[1] > converge_thresh
			counter -= 1
		counter += 1
	root_samples = [tmp[0] for tmp in root_samples_and_errors]
	errors_samples = [tmp[1] for tmp in root_samples_and_errors]
	assert max(errors_samples) <= converge_thresh

	# Add some default arguments to the error function
	from operation_table import op_table
	nr_of_funcs = len(op_table)
	nr_of_nodes = 25
	direct_error_func_curry = lambda true_root_vals, pnts, dims, cgp, nr_of_cgp_parameters, op_table: direct_error_func( start_func, input_function_and_derivatives, true_root_vals, pnts, dims, cgp, nr_of_cgp_parameters, op_table, nr_of_parameters_in_inp, nr_of_derivatives)

	# Start the optimization
	return multistart_opt(root_samples, parameter_in_inp_samples, dims, nr_of_funcs, nr_of_nodes, direct_error_func_curry, op_table, optimizer, nr_of_pars=nr_of_cgp_pars, max_iter=max_iter, multi_starts=multi_starts, max_time=max_time)
