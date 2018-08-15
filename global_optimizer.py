from random import random, randint, gauss, sample

def get_random_ind(nr_of_parameters):
	out = [gauss(0, 100) for _ in range(nr_of_parameters)]
	return out

def differential_evolution(pnts, f_vals, nr_of_parameters, objective_func, func, max_iter=100, pop_size=10):

	population = [get_random_ind(nr_of_parameters) for _ in range(pop_size)]
	errs = [objective_func(f_vals, pnts, beta, func) for beta in population]

	# Check if all errors are equal. We take that to mean that there is no use tuning.
	if min(errs) == max(errs):
		return (population[0], errs[0])

	metas = [[random(), random()] for _ in range(pop_size)]

	for itr in range(max_iter):
		for ind in range(pop_size):

			# mutate meta parameters
			if random()<0.1:
				cr = random()
			else:
				cr = metas[ind][0]
			if random()<0.1:
				f = random()
			else:
				f = metas[ind][1]

			a,b,c = sample(range(pop_size-1), 3)
			if a>=ind:
				a+=1
			if b>=ind:
				b+=1
			if c>=ind:
				c+=1

			assert not(a==b or a==c or b==c or a==ind or b==ind or c==ind)

			new_ind = list(population[ind])
			for par in range(nr_of_parameters):
				if random()<cr:
					new_ind[par] = population[a][par] + f * (population[b][par]-population[c][par])
			new_err = objective_func(f_vals, pnts, new_ind, func)
			if new_err < errs[ind]:
				metas[ind][0] = cr
				metas[ind][1] = f
				errs[ind] = new_err
				population[ind] = new_ind
	best_ind = errs.index(min(errs))

	return (population[best_ind], errs[best_ind])


