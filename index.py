from generate_root_finder_code import generate_code
from iterative_function_finder import direct_iterative_function_finder
from starting_point_approximation import starting_point_approximation

def write_2_file(s):
	import __main__
	file_name = __main__.__file__
	file_name = file_name[:-3]+".txt" # remove .py, add .txt

	import datetime
	dt_string = str(datetime.datetime.now())

	s += ";"+dt_string+"\n"

	f = open(file_name, "a")
	f.write(s)

def find_root_solver(input_function_and_derivatives, func_str, parameter_ranges_of_input_func, variable_range, optimizer, nr_of_derivatives=1):
	# TODO: We don't need both the function and its string. Fix.

	write_2_file("Starting")

	nr_of_parameters_of_input_func = len(parameter_ranges_of_input_func)
	assert len(variable_range) == 2
	assert variable_range[0] < variable_range[1]

	for par_range in parameter_ranges_of_input_func:
		assert len(par_range) == 2
		assert par_range[0] < par_range[1]

	(cgp_start_guess, error_start_guess, parameters_start_guess) = starting_point_approximation(input_function_and_derivatives[0], nr_of_parameters_of_input_func, parameter_ranges_of_input_func, optimizer, max_time=60*60*1, symbolic_der=input_function_and_derivatives[1])
	start_guess = lambda x: cgp_start_guess.eval(x, parameters=parameters_start_guess)

	write_2_file("Starting direct funtion finder")
	(cgp_iterative, error_iterative, parameters_iterative) = direct_iterative_function_finder(start_guess, input_function_and_derivatives, nr_of_parameters_of_input_func, parameter_ranges_of_input_func, variable_range, optimizer, nr_of_derivatives=nr_of_derivatives, max_iter=1000, nr_of_samples_per_inp_parameter=15, nr_of_cgp_pars=2, max_time=60*60*4)
	write_2_file("Done with direct_iterative_function_finder")
	if cgp_iterative is None:
		print("The code was unable to find a function. Please run again with longer running time.")
	assert len(parameters_start_guess) == cgp_start_guess.nr_of_parameters

	from operation_table import op_table
	print("The errors are:", error_start_guess, error_iterative)
	code = generate_code(op_table, cgp_start_guess, parameters_start_guess, cgp_iterative, parameters_iterative, nr_of_derivatives, func_str, nr_of_parameters_of_input_func)
	write_2_file("Done.")
	return code

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

	code = find_root_solver([function, der], 'x-a1*sin(x)-a2', par_ranges_of_input_func, var_range, "es")

	print(code)
	print("Total time (sec):", time() - start_time )