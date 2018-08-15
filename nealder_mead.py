from random import gauss

def calc_centroid(points, dims):
	n = len(points)-1
	centroid = [0.0]*dims

	for i in range(dims):
		for j in range(n):
			centroid[i] += points[i][j]
		centroid[i] /= float(n)
	return centroid

def nealder_mead(obj_func, dims, max_iter=50):
	points = [[gauss(0,1) for _ in range(dims)] for _ in range(dims+1)]
	point_values = [obj_func(point) for point in points]

	has_converged = False
	itr = 0

	alpha = 1.0
	gamma = 2.0
	rho = 0.5
	sigma = 0.5

	while not has_converged:
		idxs = [point_values.index(val) for val, _ in sorted(zip(point_values, points))]

		points = [points[idxs[i]] for i in range(len(idxs))]
		point_values = [point_values[idxs[i]] for i in range(len(idxs))]


		assert point_values[0]<=point_values[-1]

		centroid = calc_centroid(points, dims)
		point_reflec = [centroid[i] + alpha*(centroid[i]-points[-1][i]) for i in range(dims)]

		val = obj_func(point_reflec)

		should_shrink = True

		if val >= point_values[0] and val < point_values[dims-1-1]:
			points[-1] = point_reflec
			point_values[-1] = val
			should_shrink = False
		elif val < point_values[0]:

			point_expand = [centroid[i] + gamma*(point_reflec[i]-centroid[i]) for i in range(dims)]
			val_expand = obj_func(point_expand)

			if val_expand < val:
				points[-1] = point_expand
				point_values[-1] = val_expand
			else:
				points[-1] = point_reflec
				point_values[-1] = val
			should_shrink = False
		else:
			assert val >= point_values[dims-1-1]
			point_contract = [centroid[i] + rho*(points[-1][i] - centroid[i]) for i in range(dims)]
			val_contract = obj_func( point_contract )

			if val_contract < point_values[-1]:
				points[-1] = point_contract
				point_values[-1] = val_contract
				should_shrink = False
			else:
				assert should_shrink

		#shrink
		if should_shrink:
			for i in range(1, len(points)):
				points[i] = [points[0][j] + sigma*( points[i][j] - points[0][j]) for j in range(dims)]
				point_values[i] = obj_func(points[i])
		itr += 1


		if itr==max_iter:
			has_converged=True
	best_idx = point_values.index(min(point_values))

	return (point_values[best_idx], points[best_idx])

if __name__ == '__main__':

	from math import sin, exp, fabs

	func = lambda X: X[0]*X[0] + (sin(X[1]*X[0])*exp(X[3]))**2 + fabs(X[3]*X[2]) + X[2]*X[2]
	#func = lambda X: X[0]*X[0] + X[1]*X[1]
	print(nelder_mead(func, 4))
