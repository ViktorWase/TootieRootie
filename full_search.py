def get_all_possible_node_parts(prev_nodes, op_table):
	node_parts = []
	for op_nr in range(len(op_table)):
		if op_table[op_nr].is_binary:
			for d1 in range(prev_nodes):
				for d2 in range(prev_nodes):
					node_parts.append([op_nr, d1, d2])
		else:
			for d in range(prev_nodes):
				node_parts.append([op_nr, d, 0])
	return node_parts

def update_pos(pos, all_node_parts):
	pos[0] += 1
	for i in range(len(pos)):
		if pos[i] == len(all_node_parts[i]):
			if i == len(pos)-1:
				return False
			else:
				pos[i] = 0
				pos[i+1] += 1
		else:
			break
	return pos

def get_all_genes_sub(dims, nr_of_used_nodes, op_table):
	genes_sub = []
	print("all node parts start")
	all_node_parts = [get_all_possible_node_parts(dims+i, op_table) for i in range(nr_of_used_nodes+1)]
	print("all node parts done")
	pos = [0]*(nr_of_used_nodes+1)

	old_last_pos = pos[-1]
	while True:
		gene = []

		for i in range(nr_of_used_nodes+1):
			p = pos[i]
			assert len(all_node_parts[i][p]) == 3
			gene += all_node_parts[i][p]
		genes_sub.append(gene)

		# Update pos
		pos = update_pos(pos, all_node_parts)

		if pos is False:
			break

		if pos[-1] != old_last_pos:
			print("here (in):", pos[-1], len(all_node_parts[-1]))
			old_last_pos = pos[-1]

	return genes_sub

def nodes_used_rec(nodes_used, gene, current_node, op_table, dims):
	op = op_table[gene[current_node*3]]

	if op.is_binary:
		node_1 = gene[current_node*3+1] - dims
		if node_1 >= 0:
			nodes_used[node_1] = True
			nodes_used_rec(nodes_used, gene, node_1, op_table, dims)

		node_2 = gene[current_node*3+2] - dims
		if node_2 >= 0:
			nodes_used[node_2] = True
			nodes_used_rec(nodes_used, gene, node_2, op_table, dims)
	else:
		node = gene[current_node*3+1] - dims
		if node >= 0:
			nodes_used[node] = True
			nodes_used_rec(nodes_used, gene, node, op_table, dims)

def all_nodes_are_used(gene, nr_of_nodes, dims, op_table):
	nodes_used = [False for _ in range(nr_of_nodes)]
	first_node = gene[-1]-dims

	if first_node >= 0:
		nodes_used[first_node] = True
		nodes_used_rec(nodes_used, gene, first_node, op_table, dims)
		return all(nodes_used)
	else:
		return False

def get_all_genes(dims, nr_of_nodes, direct_error_func_curry, root_samples, parameter_in_inp_samples, nr_of_parameters_in_inp, op_table):
	from operation_table import op_table
	genes = []

	gene_len = nr_of_nodes*3+1

	def err_func(gene):
		cgp = CGP(dims, op_table, gene)
		(err, pars) = direct_error_func_curry(root_samples, parameter_in_inp_samples, dims, cgp, nr_of_parameters_in_inp, op_table)

		return err
	best_err = 1.0e20
	best_gene = None

	# First we create genes that take the one of the inputs as output
	for d in range(dims):
		gene = [0]*gene_len
		gene[-1] = d

		if all_nodes_are_used(gene, nr_of_nodes, dims, op_table):
			err = err_func(gene)
			if err < best_err:
				best_err = err
				best_gene = list(gene)
				print("best", err)

	# And then the other ones
	for node in range(nr_of_nodes):
		print("Start")
		subs = get_all_genes_sub(dims, node, op_table)
		print("mid")
		len_of_sub = len(subs[0])
		remaining_len = gene_len-len_of_sub
		assert remaining_len >= 0
		if remaining_len != 0:
			for i in range(len(subs)):
				subs[i] = subs[i] + [0]*remaining_len
				subs[i][-1] = dims + node

				if all_nodes_are_used(subs[i], nr_of_nodes, dims, op_table):
					err = err_func(subs[i])
					if err < best_err:
						best_err = err
						best_gene = list(subs[i])
						print("best", err, "   progress:", float(i)/len(subs))
				subs[i] = None
				if i%50000 == 0:
					print("here", node, nr_of_nodes)
					print("itr", i, len(subs))
			subs = None
	return genes

def full_search(input_function_and_derivatives, nr_of_parameters_in_inp, parameter_ranges, variable_range, nr_of_samples_per_inp_parameter=5, nr_of_nodes=3, nr_of_variable_samples=15):
	func = input_function_and_derivatives[0]

	# The dimensions of the cgp are x, the value of the given function and its derivatives, as well
	# as the parameters of the input function.
	dims = 1 + nr_of_derivatives+1+nr_of_parameters_in_inp
	#parameter_ranges = [[0.0, 0.1], [0.0, 2*pi]]
	#variable_range = [0.0, 2*pi]

	#nr_of_nodes = 3

	#nr_of_samples_per_inp_parameter = 5
	#nr_of_variable_samples = 15

	(x_pnts, parameter_in_inp_samples, nr_of_par_in_inp_samples) = get_parameter_pnts(nr_of_parameters_in_inp, parameter_ranges, variable_range, nr_of_samples_per_inp_parameter, nr_of_variable_samples)

	# Let's get the derivative as well. TODO: this shouldn't have to be numerical.
	func_der = lambda x, a: (func([x[0]+1.0e-11] , a) - func([x[0]],a))/1.0e-11 # TODO: Do this symbolically, not numerically.
	# Find the root values using Newton-Raphson. TODO: Add a bunch of root finders here to make sure that it converges.
	root_samples_and_errors = [newton_raphson(func, func_der, parameter_in_inp_samples[i]) for i in range(nr_of_par_in_inp_samples)] # TODO: Change from NR to a lot of root solvers

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
	start_func = lambda pars: 0.0 # TODO: Find the start func instead

	direct_error_func_curry = lambda true_root_vals, pnts, dims, cgp, nr_of_cgp_parameters, op_table: direct_error_func( start_func, input_function_and_derivatives, true_root_vals, pnts, dims, cgp, nr_of_cgp_parameters, op_table, nr_of_parameters_in_inp, nr_of_derivatives)

	print("Dims:", dims)
	genes = get_all_genes(dims, nr_of_nodes, direct_error_func_curry, root_samples, parameter_in_inp_samples, 2, op_table)
	print(len(genes))

if __name__ == '__main__':
	from operation_table import op_table
	from math import pi, sin, cos
	from cgp import CGP

	from newton_raphson import newton_raphson
	from iterative_function_finder import get_parameter_pnts, direct_error_func
	
	function = lambda x, P: x[0]-P[0]*sin(x[0])-P[1]
	der = lambda x, P: 1-P[0]*cos(x[0])

	nr_of_derivatives = 1
	nr_of_parameters_in_inp = 2

	var_range = [0.0, 2*3.141592]
	par_ranges_of_input_func = [[0.0, 0.5], [0.0, 3.141592]]
	input_function_and_derivatives = [function, der]

	full_search(input_function_and_derivatives, nr_of_parameters_in_inp, par_ranges_of_input_func, var_range)

	
	"""

	best_err = 1.0e20
	n = len(genes)
	for i in range(n):
		gene = genes[0]
		cgp = CGP(dims, op_table, gene)

		(err, pars) = direct_error_func_curry( root_samples, parameter_in_inp_samples, dims, cgp, 2, op_table)
		if err < best_err:
			print(err, i, len(genes))
			best_err = err

		if i%3000 == 0:
			print("Progress", round(float(i)/len(genes) * 100, 3), "%")

		genes.pop(0)
	print("Best err", best_err)


	"""
