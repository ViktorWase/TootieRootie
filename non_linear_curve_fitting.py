from sys import version_info
if version_info >= (3,0):
	from math import sqrt, inf, floor
else:
	from math import sqrt, floor
	inf = 1.0e25

from global_optimizer import differential_evolution

def mat_vec_mult(A, v):
	"""
	Multiplies a vector v with a matrix A.
	"""
	m = len(v)
	assert(m == len(A[0]))
	n = len(A)
	return [sum(A[i][j]*v[j] for j in range(m)) for i in range(n)]

def gradient_descent(residual_func, f_vals, nr_of_parameters, nr_of_pnts, pnts, jacobian_function, func, max_iter=500, start_val=None, rel_thresh=1.0e-9, terminal_cond=1.0e-8):
	# TODO: Add stepsize increase as well

	if start_val == None:
		beta = [1.0e-8] * nr_of_parameters
	else:
		beta = list(start_val)

	step_size = 1.0e-3 # TODO: FIX!
	#print("Par nr:", nr_of_parameters)

	"""
	old_err = inf
	for itr in range(max_iter):
		improved=False
		differences = residual_func(f_vals, pnts, beta, func)
		
		#err = sqrt(sum(d*d for d in differences))
		err = sum(d*d for d in differences)

		if err >= old_err:
			# Undo last step
			beta = [beta[i] + grad[i]*step_size for i in range(nr_of_parameters)]

			# And shrink the step size
			step_size /= 2.0
		else:
			jacobian = jacobian_function(f_vals, pnts, beta, func)
			grad = [sum(jacobian[j][i]*differences[j] for j in range(nr_of_pnts)) for i in range(nr_of_parameters)]

			#step_size *= 1.5
			old_err = err
			improved = True
		if itr != max_iter-1:
			beta = [beta[i]-grad[i]*step_size for i in range(nr_of_parameters)]
	"""

	differences = residual_func(f_vals, pnts, beta, func)
	error_of_fixed_point = sum(d*d for d in differences)
	fixed_point = list(beta)
	#print("----------------")
	has_converged = False
	for itr in range(max_iter):
		#print("---")
		#differences = residual_func(f_vals, pnts, fixed_point, func)
		jacobian = jacobian_function(f_vals, pnts, fixed_point, func)
		#print("jac", jacobian)
		#print("diff", differences)
		grad = [sum(jacobian[j][i]*differences[j] for j in range(nr_of_pnts)) for i in range(nr_of_parameters)] # TODO: Irrelevant I suppose, but I think this one should be multiplied by 2?

		# Decide step size
		is_first_iter = True
		while True:
			beta = [fixed_point[i] - step_size*grad[i] for i in range(nr_of_parameters)]
			differences = residual_func(f_vals, pnts, beta, func)
			err = sum(d*d for d in differences)

			if err < error_of_fixed_point:
				fixed_point = list(beta) #TODO: Is this list necessary?
				error_of_fixed_point = err
				if is_first_iter:
					step_size *= 1.5
				break
			else:
				#print(err, error_of_fixed_point)
				if step_size < 1.0e-14:
					has_converged = True
					break
				is_first_iter = False
				step_size /= 2.0

		if has_converged:
			#print("Converged after", itr)
			break

	return (fixed_point, error_of_fixed_point)

def combo_curve_fitting(residual_func, f_vals, nr_of_parameters, nr_of_pnts, pnts, jacobian_function, func, max_iter=500):

	downsampling_factor = 10 # TODO: Make this an input
	if nr_of_pnts > 60:
		pnts_downsampled = [pnts[downsampling_factor*i] for i in range(int(floor(nr_of_pnts/downsampling_factor)))]
	else:
		pnts_downsampled = pnts

	objective_func = lambda f_vals, pnts, beta, func: sum(d*d for d in residual_func(f_vals, pnts, beta, func))
	(pars_de, err_de_downsampled) = differential_evolution(pnts_downsampled, f_vals, nr_of_parameters, objective_func, func, max_iter=5)

	err_de = sum(d*d for d in residual_func(f_vals, pnts, pars_de, func))

	(pars, err_gd) = gradient_descent(residual_func, f_vals, nr_of_parameters, nr_of_pnts, pnts, jacobian_function, func, start_val=pars_de, max_iter=max_iter)

	if err_de < err_gd:
		#print("Gradient descent fucked up", err_de, err_gd, err_gd-err_de)
		err = err_de
		pars = pars_de
	else:
		#print("Grad desc improved by", err_de-err_gd)
		err = err_gd
	#print("curve fitting:", pars, err)

	return (pars, err)
