"""
Simulated anneling is an optimization method that we use to find suitable
functions. Note that it is used for discrete optimization only. 
The countinous optimization (that is used to find the correct values of the
numerical parameters/constants in the function) is Gauss-Newton, and it is in
another file.
"""

def write_2_file(pars, itr, current_error, optimizer, cgp_str): # TODO: This function exists in index. Put it in a separate file and use for both.
	import datetime
	dt_string = str(datetime.datetime.now())
	import __main__
	file_name = __main__.__file__
	file_name = file_name[:-3]+".txt" # remove .py, add .txt
	s = ""
	for p in pars:
		s += str(p)+","
	s+=";"+str(current_error)+";"+str(itr)+";"+optimizer+";"+cgp_str+";"+dt_string+"\n"
	f = open(file_name, "a")
	f.write(s)


from copy import copy, deepcopy
from random import random, randint
from time import time

from sys import version_info
if version_info >= (3,0):
	from math import sqrt, inf, exp
else:
	from math import sqrt, exp
	inf = 1.0e25

from cgp import Operation, CGP

def create_random_gene(dims, nr_of_funcs, nr_of_nodes):
	"""
	Creates a random gene that represents a random cgp function.
	Assumes that the operations are unary or binary.
	"""
	n = 3*nr_of_nodes + 1
	counter = 0
	gene = [None for _ in range(n)]
	for i in range(nr_of_nodes):
		gene[counter] = randint(0, nr_of_funcs - 1)
		counter += 1

		for _ in range(2):
			gene[counter] = randint(0, i+dims - 1)
			counter += 1
	gene[counter] = randint(0, nr_of_nodes-1+dims)
	return gene

def mutate(cgp_in, dims, nr_of_funcs, mute_rate=0.4):
	"""
	Mutates the cgp. Doesn't affect the input CGP.

	One of the USED parts of the gene is always mutated.
	"""
	gene = list(cgp_in.gene)
	nodes = int((len(gene)-1)/3)

	nr_of_used_parts =  sum(cgp_in.used_genes)+1 # The +1 is for the last number in the gene, which decides which node that is the output. THe -1 is for indexing.
	used_part_2_mutate = 0 if nr_of_used_parts<=0 else randint(0, nr_of_used_parts-1)
	used_part_counter = 0
	counter = 0
	has_forced_mutated = False
	for i in range(nodes):

		# Make sure at least some part of the USED gene is mutated...
		# TODO: REMOVE THE WHOLE THING WHER EIT MUTATES INTO SOMETHING NEW. It can mutate into itself.
		if cgp_in.used_genes[i]:
			if used_part_counter == used_part_2_mutate:
				assert has_forced_mutated == False
				has_forced_mutated = True
				is_binary = cgp_in.op_table[gene[counter]].is_binary

				random_node = randint(0, 2 + (1 if is_binary else 0)-1)
				if random_node == 0:
					has_changed = False
					while not has_changed:
						old_val = gene[counter]
						gene[counter] = randint(0, nr_of_funcs-1)
						if gene[counter] != old_val:
							has_changed = True
				else:
					assert random_node==1 or random_node==2
					has_changed = False
					while not has_changed:
						old_val = gene[counter+random_node]
						gene[counter+random_node] = randint(0,i+dims - 1)
						if gene[counter+random_node] != old_val:
							has_changed = True
			#counter += 3
			used_part_counter += 1

		
		#... the other parts don't have to mutate
		if random()<mute_rate:
			gene[counter] = randint(0, nr_of_funcs-1)
		counter += 1

		for _ in range(2):
			if random()<mute_rate:
				gene[counter] = randint(0,i+dims - 1)
			counter += 1
	if random() < mute_rate or nr_of_used_parts-1 == used_part_2_mutate:
		assert counter == len(gene)-1
		if nr_of_used_parts-1 == used_part_2_mutate:
			assert has_forced_mutated == False
		has_forced_mutated = True
		old_val = gene[counter]
		gene[counter] = randint(0, nodes-1+dims)
	assert has_forced_mutated

	# This tmp is just to make sure it does not cause a seg-fault.
	start_node = gene[-1]
	tmp = gene[3*(start_node-dims)]

	assert counter == len(gene)-1
	assert dims == cgp_in.dims+cgp_in.nr_of_parameters
	return CGP(cgp_in.dims, cgp_in.op_table, gene, nr_of_parameters=cgp_in.nr_of_parameters)

def acceptance_prob(new_error, current_error, temp):
	"""
	The acceptance probability depends on the difference in the objective
	function as well as the "temperature" of the system. Higher temperature means that anything could happen.
	"""
	diff = new_error - current_error
	return exp(-(diff)/temp)

def sa(f_vals, pnts, dims, nr_of_funcs, nr_of_nodes, error_func, op_table, max_iter=500, nr_of_pars=0, reheat_iter=100, remaining_time=None):
	"""
	Simulated anneling is a simple way of doing compinatorial optimization without getting stuck in local minima.
	It basically works like this: 

	1) take the current solution and apply a small change to it

	2) If this new solution is better, keep it.

	3) There is a chance that the new solution is kept even if it is worse.
	   This chance decreases as the iteration number grows, and the chance
	   is small if the new solution is much worse than the old solution.

	4) Repeate the process a bit, and return the best solution.

	The wiki page is rather good as well.
	"""

	start_time = time()

	assert nr_of_funcs == len(op_table)
	# Create a starting function (solution) at random.
	current_sol = create_random_gene(dims+nr_of_pars, nr_of_funcs, nr_of_nodes)

	current_cgp = CGP(dims, op_table, current_sol, nr_of_parameters=nr_of_pars)

	(current_error, best_pars) = error_func(f_vals, pnts, dims, current_cgp, nr_of_pars, op_table)

	print(current_error)
	#assert False

	best_cgp = deepcopy(current_cgp)
	best_error = current_error

	iterations_since_update = 0

	temperature_itr = 0

	for itr in range(max_iter):
		if itr % 50 == 0:
			print("iter:", itr," of", max_iter)
		temp = float(max_iter-temperature_itr)/max_iter
		# Do a small mutation to create a new function (aka solution)
		new_cgp = mutate(current_cgp, dims+nr_of_pars, nr_of_funcs)
		#cgp = CGP(dims, op_table, new_sol, nr_of_parameters=nr_of_pars)
		(new_error, new_pars) = error_func(f_vals, pnts, dims, new_cgp, nr_of_pars, op_table)

		temperature_itr += 1


		if new_error < current_error or acceptance_prob(new_error, current_error, temp)<random():
			#current_sol = new_sol
			current_cgp = new_cgp
			current_error = new_error

			if new_error < best_error:
				print("best yet:", new_error)
				write_2_file(new_pars, itr, current_error, 'sa',current_cgp.convert2str(parameters=new_pars))
				new_cgp.print_function(parameters=new_pars)
				best_cgp = deepcopy(new_cgp)
				best_error = new_error
				best_pars = list(new_pars)
		else:
			iterations_since_update += 1

			# If no change has been made in a while, then we set the temp to max again!
			if iterations_since_update == reheat_iter:
				temperature_itr = 0
				iterations_since_update = 0
				print("Reheating.")

		if remaining_time!=None and time()-start_time >= remaining_time:
			break

	return (best_cgp, best_error, best_pars)


def es(f_vals, pnts, dims, nr_of_funcs, nr_of_nodes, error_func, op_table, max_iter=500, nr_of_pars=0, reheat_iter=100, remaining_time=None, nr_of_children=4):
	"""
	Evolutionary Strategy.
	"""

	start_time = time()

	assert nr_of_funcs == len(op_table)
	# Create a starting function (solution) at random.
	current_sol = create_random_gene(dims+nr_of_pars, nr_of_funcs, nr_of_nodes)

	current_cgp = CGP(dims, op_table, current_sol, nr_of_parameters=nr_of_pars)

	(current_error, best_pars) = error_func(f_vals, pnts, dims, current_cgp, nr_of_pars, op_table)

	print(current_error)

	best_cgp = deepcopy(current_cgp)
	best_error = current_error

	for itr in range(max_iter):
		if itr % 50 == 0:
			print("iter:", itr," of", max_iter)
		
		# Do a small mutation to create a few new function (aka solution)
		children = [mutate(current_cgp, dims+nr_of_pars, nr_of_funcs) for _ in range(nr_of_children)]

		children_results = [error_func(f_vals, pnts, dims, child, nr_of_pars, op_table) for child in children]
		children_errors = [children_results[i][0] for i in range(nr_of_children)]
		children_pars = [children_results[i][1] for i in range(nr_of_children)]

		best_children_idx = children_errors.index(min(children_errors))

		current_cgp = children[best_children_idx]
		current_error = children_errors[best_children_idx]

		if current_error < best_error:
			new_pars = children_pars[best_children_idx]
			print("best yet:", current_error)

			write_2_file(new_pars, itr, current_error, 'es',current_cgp.convert2str(parameters=new_pars))
			current_cgp.print_function(parameters=new_pars)
			best_cgp = deepcopy(current_cgp)
			best_error = current_error
			best_pars = list(new_pars)

		if remaining_time!=None and time()-start_time >= remaining_time:
			break

	return (best_cgp, best_error, best_pars)

def multistart_opt(f_vals, pnts, dims, nr_of_funcs, nr_of_nodes, error_func, op_table, optimizer, max_iter=1000, multi_starts=10, nr_of_pars=0, max_time=None):
	"""
	A multistart version of simulated anneling/es. Returns the best found solution.
	"""
	best_err = inf
	best_sol = None
	best_pars = None

	if max_time != None:
		multi_starts = None

	counter = 0

	start_time = time()
	remaining_time = True

	while True:
		print("STARTING NEW:",counter+1,"of", multi_starts )

		if max_time != None:
			passed_time = time() - start_time
			remaining_time = max_time - passed_time
		if optimizer=="sa":
			(sol, err, pars) = sa(f_vals, pnts, dims, nr_of_funcs, nr_of_nodes, error_func, op_table, max_iter=max_iter, nr_of_pars=nr_of_pars, remaining_time=remaining_time)
		elif optimizer=="es":
			(sol, err, pars) = es(f_vals, pnts, dims, nr_of_funcs, nr_of_nodes, error_func, op_table, max_iter=max_iter, nr_of_pars=nr_of_pars, remaining_time=remaining_time)
		else:
			print(optimizer, "is not a used optimizer.")
			assert False

		if err < best_err:
			best_err = err
			best_sol = sol
			best_pars = pars
		assert len(pars) == nr_of_pars

		counter += 1

		if multi_starts != None:
			if counter == multi_starts:
				break
		else:
			if time() - start_time >= max_time:
				print("STOPPING")
				break
			else:
				print(time() - start_time, max_time)
	return (best_sol, best_err, best_pars)


if __name__ == '__main__':
	from math import cos
	op_table = [Operation("+"), Operation("*"), Operation("sin"), Operation("cos"), Operation("sqr"), Operation("-"), Operation("log"), Operation("/")]

	def err_func(outs, inps, dims, gene, nr_of_pars, op_table):
		"""
		Example of an error function, without parameters or anything fancy. 
		Just a minimization of the L2-norm.
		"""
		dims = len(inps[0])
		cgp = CGP(dims, op_table, gene)
		n = len(inps)
		assert len(outs) == n

		s = 0.0
		for i in range(n):
			tmp = outs[i] - cgp.eval(inps[i])
			s += tmp*tmp
		s /= float(n)
		return sqrt(s)

	def f(x):
		return cos(x[1])*0.9 + x[0]*x[0]*1.1

	inputs = [[random(), random()] for _ in range(150)]
	outputs = [f(x) for x in inputs]


	nr_of_nodes = 15
	print(multistart_opt(outputs, inputs, len(inputs[0]), len(op_table), nr_of_nodes, err_func, op_table))
