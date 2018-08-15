# TODO: This is all sympy is used for now. Remove? Or maybe add some symbolic derivation in the symbolic regression.
from sympy.parsing.sympy_parser import parse_expr
from sympy import diff, simplify

from cgp import convert
#from nonlinear_system_of_equation_solver import get_iterative_improvement_function

def change_symbols_2_parameters(in_str, nr_of_parameters_in_input_func):
	for i in range(nr_of_parameters_in_input_func):
		in_str = in_str.replace("a"+str(i+1), "pars["+str(i)+"]")
	return in_str

def generate_code(op_table, first_approx_cgp, first_approx_parameters, improvement_cgp, improvement_parameters, derivatives_used, func_str, nr_of_parameters_in_input_func):
	code = "from math import sin, cos, tan, exp, log, sqrt, pow\n" # TODO: Make sure all relevant functions are imported and one else.

	sympy_func = parse_expr(func_str)

	# TODO: A lot of this is copy-pasted from the automatic derivation function.
	derivatives_and_func = [sympy_func]
	current_derivative = derivatives_and_func[0]
	for i in range(derivatives_used):
		current_derivative = diff(current_derivative, 'x')
		current_derivative = simplify(current_derivative)
		derivatives_and_func.append(current_derivative)

	func_str = change_symbols_2_parameters(func_str, nr_of_parameters_in_input_func)

	code += "def function(x, pars):	\n	return "+func_str+"\n\n"

	code += "def derivative(x, pars, derivative_number):\n"
	for i in range(derivatives_used):
		if i==0:
			code += "	if derivative_number==1:\n"
		else:
			code += "	elif derivative_number=="+str(i+1)+":\n"
		code += "		return "+change_symbols_2_parameters(str(derivatives_and_func[i+1]), nr_of_parameters_in_input_func)+"\n"
		code += "\n"

	code += "def starting_approximation(parameters):"+"\n"+"	return "+convert(op_table, first_approx_cgp.gene, ['parameters['+str(i)+']' for i in range(first_approx_cgp.dims)], first_approx_cgp.nr_of_nodes, first_approx_cgp.dims, parameters=first_approx_parameters)

	#improvement_func_sympy = get_iterative_improvement_function(improvement_cgp, derivatives_used)
	improvements_func = convert(op_table, improvement_cgp.gene, ['parameters['+str(i)+']' for i in range(improvement_cgp.dims)], improvement_cgp.nr_of_nodes, improvement_cgp.dims, parameters=improvement_parameters)

	code += "\n\ndef iterative_root_finding(c, pars):	\n"

	line = "	k1 = function(c, pars)\n"
	code += line
	for i in range(derivatives_used):
		line = "	k"+str(i+1+1)+" = derivative(c, pars, "+str(i+1)+")\n"
		code += line

	code += "	return "+str(improvements_func)+"\n"

	code += """
def root_finder(parameters, max_iter=500):
	x = starting_approximation(parameters)

	for i in range(max_iter):
		x = iterative_root_finding(x, parameters)
		# TODO: Insert a threshold
	return x
"""

	return code

if __name__ == '__main__':
	from cgp import CGP, Operation

	op_table = [Operation("+"), Operation("*"), Operation("sin"), Operation("cos"), Operation("sqr"), Operation("-"), Operation("log"), Operation("/")]	

	gene = [1,2,0,3]
	first_approx_cgp = CGP(2, op_table, gene, nr_of_parameters=1)

	nr_of_der = 1
	gene2 = [1,1,0  ,0,3,2 ,4]
	improvement_cgp =  CGP(1, op_table, gene2, nr_of_parameters=nr_of_der+1)


	func_str = 'a1*sin(x)+a2*x+a3'

	print(convert(op_table, improvement_cgp.gene, ['x'], improvement_cgp.nr_of_nodes, improvement_cgp.dims, parameters=[0.1, 0.4]))
	print(generate_code(op_table, first_approx_cgp, [0.4], improvement_cgp, nr_of_der, func_str, 3))

	
	