from index import find_root_solver

if __name__ == '__main__':
	from math import sin, pi, cos
	from random import seed
	from time import time

	start_time = time()

	seed(2)
	function = lambda x, P: x[0]-P[0]*sin(x[0])-P[1]
	der = lambda x, P: 1-P[0]*cos(x[0])

	var_range = [0.0, 2*3.141592]
	par_ranges_of_input_func = [[0.0, 0.5], [0.0, 3.141592]]

	code = find_root_solver([function, der], 'x-a1*sin(x)-a2', par_ranges_of_input_func, var_range)

	print(code)

	print("Total time (sec):", time() - start_time )
	print(1)