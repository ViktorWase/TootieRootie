from index import find_root_solver

if __name__ == '__main__':
	from math import sin, pi
	from random import seed
	from time import time

	start_time = time()

	seed(2)
	"""
	function = lambda x, P: P[0]*x*x*x + P[1]*x*x + P[2]*x + P[3]
	der = lambda x, P: 3*P[0]*x*x + 2*P[1]*x + P[2]
	derder = lambda x, P: 6*P[0]*x + 2*P[1]

	var_range = [0.0, 1.0]
	par_ranges_of_input_func = [[0.0, 1.0], [0.0, 1.0] ,[0.0, 1.0], [0.0, 1.0]]
	"""

	function = lambda x, P: x[0]*x[0]-P[0]
	der = lambda x, P: 2*x[0]

	var_range = [0.0, 1.0]
	par_ranges_of_input_func = [[0.0, 10.0]]
	

	code = find_root_solver([function, der], 'x*x-a1', par_ranges_of_input_func, var_range)

	print(code)

	print("Total time (sec):", time() - start_time )
	print(3)