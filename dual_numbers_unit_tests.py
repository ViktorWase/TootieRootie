from cgp import CGP, Operation
from simulated_anneling import create_random_gene

from random import randint, gauss
from math import sqrt, cos, sin

# Test 1
from dual_numbers import cosd, logd, DualNumber
f = lambda x: cosd(x)*x+DualNumber(1.5,0.0)*x
f_der = lambda x:cos(x)-sin(x)*x + 1.5
err = 0.0
for _ in range(50):
	x = gauss(0, 10)
	d_der = f(DualNumber(x, 1.0))
	der = f_der(x)

	err += (d_der.b-der)*(d_der.b-der)
print("Test 1 error:", err)


# Test 2

# Create a random cgp
op_table = [Operation("+"), Operation("*"), Operation("sin"), Operation("cos"), Operation("sqr"), Operation("-"), Operation("log"), Operation("/")]

h = 1.0e-9

err = 0.0
counter = 0.0
for _ in range(10):
	nr_of_nodes = randint(1, 15)
	dims = randint(1, 4)
	gene = create_random_gene(dims, len(op_table), nr_of_nodes)

	cgp = CGP(dims, op_table, gene, nr_of_parameters=0)

	for _ in range(10):
		pnt = [gauss(0,10) for _ in range(dims)]

		for d in range(dims):
			pnt_shift = list(pnt)
			pnt_shift[d] += h
			numerical_der = (cgp.eval(pnt_shift) - cgp.eval(pnt))/h
			analytical_der = cgp.eval(pnt, derivative=True, der_dir=d)
			diff = analytical_der[1] - numerical_der

			err += diff*diff
			counter += 1.0
err = sqrt(err/counter)
print("Test 2 error:", err)

print("Both errors should be below 1.0e-5")