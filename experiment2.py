from index import find_root_solver

if __name__ == '__main__':
	from math import sin, pi, sqrt, acos
	from random import seed
	from time import time

	start_time = time()

	seed(2)
	function = lambda x, P: acos(1.0/ ( x[0]/P[0] - 1.0))+acos(1.0/ ( x[0]/P[1] - 1.0))-P[2] if x[0] >= 0 else 1.0e10
	der = lambda x, P: P[0] /( (x[0]-P[0])*(x[0]-P[0])*sqrt(1.0 - (P[0]/(x[0]-P[0]))**2))+P[1]/( (x[0]-P[1])*(x[0]-P[1])*sqrt(1.0 - (P[1]/(x[0]-P[1]))**2)) if x[0] >= 0 else 0.0

	var_range = [0.0, 1.0]
	par_ranges_of_input_func = [[-1.0, 0.0], [-1.0, 0.0] ,[pi, 2*pi]]

	code = find_root_solver([function, der], 'acos(1.0/ ( x/a1 - 1.0))+acos(1.0/ ( x/a2 - 1.0))-a3', par_ranges_of_input_func, var_range)

	print(code)

	print("Total time (sec):", time() - start_time )
	print(2)


