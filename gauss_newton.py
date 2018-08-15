"""
When we want to find the values of the numerical parameters/constants we
use Gauss-Newton. That is all that is in this file.

We might need to use some global optimization (I'm voting for
 self-adaptive differential evolution) later. 
"""

import numpy as np # TODO: Should we actually use numpy?

from sys import version_info
if version_info >= (3,0):
	from math import sqrt, inf, fabs
else:
	from math import sqrt, fabs
	inf = 1.0e25


def mat_vec_mult(A, v):
	"""
	Multiplies a vector v with a matrix A.
	"""
	m = len(v)
	assert(m == len(A[0]))
	n = len(A)
	return [sum(A[i][j]*v[j] for j in range(m)) for i in range(n)]


def gauss_newton(residual_func, f_vals, nr_of_parameters, nr_of_pnts, pnts, jacobian_function, func, max_iter=100, start_val=None, rel_thresh=1.0e-9, terminal_cond=1.0e-8):
	"""
	Does Guass Newton iteration in order to 
	decide the parameter values that minimizes
	the square sum of the differences in the
	function points.
	"""
	print("GAUSS")
	assert nr_of_parameters > 0

	err = -1
	min_iter = 5

	# An approximate size of the function values.
	magnitude = sum(fabs(f) for f in f_vals) # TODO: This shouldn't have to be computed every time.

	abs_thresh = rel_thresh*magnitude

	# It shouldn't be lower than machine pres
	if abs_thresh < 1.0e-13:
		abs_thresh = 10e-13

	if start_val == None:
		beta = [1.0e-8] * nr_of_parameters
	else:
		beta = list(start_val)

	assert (nr_of_pnts >= nr_of_parameters )

	old_error = inf

	for itr in range(max_iter):
		differences = residual_func(f_vals, pnts, beta, func)

		# The error is the L2-norm of the residual errors
		err = 0.0
		for i in range(len(differences)):
			err += differences[i]*differences[i]
		err /= float(len(differences))
		err = sqrt(err)

		if( err < abs_thresh or (itr>=min_iter and fabs((err-old_error)/old_error) < 1.0e-10) ):
			print("case 1 - ", itr)
			break

		# If the error gets worse, then we undo it and take a small step 
		# gradient decsent instead (the error function is the L2-norm 
		# of the residual vector).
		if err > old_error and itr>0:
			for i in range(nr_of_parameters):
				beta[i] += delta[i]

			grad_delta = [0.0]*nr_of_parameters
			for i in range(nr_of_parameters):
				grad_delta[i] = 2*sum(differences_old[j] * jacobian[j][i] for j in range(nr_of_pnts))
				grad_delta[i] *= 0.05
				beta[i] -= grad_delta[i]

			differences = residual_func(f_vals, pnts, beta, func)
			tmp_err = sqrt(sum(d*d for d in differences)/float(len(differences)))

		jacobian = jacobian_function(f_vals, pnts, beta, func)
		#inv_jacobian = moore_penrose_pseudoinverse(jacobian)
		if nr_of_parameters>1:
			inv_jacobian = np.linalg.pinv(jacobian)
		elif nr_of_parameters==1:
			# The special case of the psuedo inverse in 1 dim and 1 parameter
			# TODO: I bet this can be sped up.
			mag_square = sum(jacobian[i][0]*jacobian[i][0] for i in range(len(jacobian)))
			if mag_square == 0:
				inv_jacobian = [[0.0 for _ in range(len(jacobian))]]
			else:
				inv_jacobian = [[jacobian[i][0]/mag_square for i in range(len(jacobian))]]
		else:
			assert False

		delta = mat_vec_mult(inv_jacobian, differences)
		for i in range(nr_of_parameters):
			beta[i] -= delta[i]

		"""
		# If there's no improvement, then we'll stop.
		if sqrt(sum(d*d for d in delta)) < terminal_cond and itr>min_iter:
			err = sqrt(sum(d*d for d in differences)/float(len(differences)))
			print("case 2 - ", itr)
			break
		"""

		differences_old = list(differences)
		old_error = err
	assert(err!=-1)
	err = sqrt(sum(d*d for d in differences)/float(len(differences)))
	return (beta, err)


if __name__ == '__main__':
	# TODO: Replace the jacobian and residual with proper and general functions later
	from random import random, gauss, seed
	from math import sin, sqrt, cos

	seed(0)

	def func(parameters, x):
		return parameters[0] * cos(parameters[1] * x[0]) + parameters[2]*x[0] + parameters[1]*0.4

	def func_der(parameters, x, der_wrt_parameter_nr):
		if der_wrt_parameter_nr == 0:
			return sin(parameters[1] * x[0])
		elif der_wrt_parameter_nr == 1:
			return parameters[0] * x[0] * cos(parameters[1] * x[0])
		elif der_wrt_parameter_nr == 2:
			return x[1]*x[1]
		else:
			assert( False )

	def residual_der(f_val, parameters, x, der_wrt_parameter_nr):
		return 2.0*(func_der(parameters, x, der_wrt_parameter_nr)-f_val)

	def res_func(f_vals, pnts, beta, func):
		nr_of_pnts = len(pnts)
		differences = [0.0] * nr_of_pnts
		for i in range(nr_of_pnts):
			tmp = func(beta, pnts[i]) - f_vals[i]
			differences[i] = tmp
		return differences

	def jacobian_function(f_vals, pnts, beta, func):
		"""
		Approximates the jacobian function using numerical derivation.
		This will be replaced with automatical derivation using dual
		numbers later.
		"""
		h = 1.0e-8
		res_base = res_func(f_vals, pnts, beta, func)
		nr_of_residuals = len(res_base)
		nr_of_pars = len(beta)
		beta_copy = list(beta)
		nr_of_pnts = len(pnts)

		jacobian = [[0.0 for i in range(nr_of_pars)] for j in range(nr_of_residuals)]

		for i in range(nr_of_pars):
			beta_copy[i] += h
			res = res_func(f_vals, pnts, beta_copy, func)
			for j in range(nr_of_residuals):
				jacobian[j][i] = (res[j]-res_base[j])/h
			beta_copy[i] -= h
		return jacobian

	pars = [0.1, 0.6, 0.04]

	pnts = [[random()] for _ in range(6)]
	f_vals = [func(pars, pnt)+gauss(0, 1.0e-3) for pnt in pnts]

	print(gauss_newton(res_func, f_vals, 3, len(pnts), pnts, jacobian_function, func))