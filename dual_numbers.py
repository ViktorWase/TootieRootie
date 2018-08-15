from math import sin, cos, exp, log, sqrt, asin, acos
from sys import version_info

class DualNumber():
	"""
	A dual number is a+b*e, where e*e=0. It's very useful for automatic differentiation.
	"""
	def __init__(self, a, b):
		self.a = a
		self.b = b
	
	# Overloads the standard operations: +, -, *, /.
	def __add__(self, other):
		return DualNumber(self.a+other.a, self.b+other.b)

	def __sub__(self, other):
		return DualNumber(self.a-other.a, self.b-other.b)

	def __mul__(self, other):
		return DualNumber(self.a*other.a, self.b*other.a + self.a*other.b)

	if version_info >= (3,0):
		def __truediv__(self, other):
			return DualNumber(self.a/other.a, (self.b*other.a-self.a*other.b)/(other.a*other.a))
	else:
		def __div__(self, other):
			return DualNumber(self.a/other.a, (self.b*other.a-self.a*other.b)/(other.a*other.a))

	def __str__(self):
		if self.b>=0:
			return str(self.a)+"+"+str(self.b)+"e"
		else:
			return str(self.a)+""+str(self.b)+"e"

# Standard functions for dual numbers
def sind(x):
	return DualNumber(sin(x.a), x.b*cos(x.a))		

def cosd(x):
	return DualNumber(cos(x.a), -x.b*sin(x.a))

def expd(x):
	return DualNumber(exp(x.a), x.b*exp(x.a))

def logd(x):
	return DualNumber(log(x.a), x.b/x.a)

def sqrtd(x):
	return DualNumber(sqrt(x.a), x.b*0.5/sqrt(x.a))

def powd(x, k):
	return DualNumber(pow(x.a, k), k*pow(x.a, k-1.0)*x.b)

def acosd(x):
	return DualNumber(acos(x.a), -x.b/sqrt(1.0-x.a*x.a))

def asind(x):
	return DualNumber(asin(x.a), x.b/sqrt(1.0-x.a*x.a))

if __name__ == '__main__':
	x = DualNumber(1.0, 1.0)
	print(cosd(x))

	from math import cos
	print((cos(1.0001)-cos(1.0))/0.0001)