# Tootie Rootie
A fixed point root finder is a simple thing, really. It is given an equation f(x)=0 and an initial guess x_0, as well as an "update function" g(x). Then it creates iterative approximations of the root by x_{i+1}:= g(x_i). A well-known example of this is the Newton-Raphson algorithm where g(x) := x-f(x)/f'(x).

Tootie Rootie creates tailor-made fixed-point root finders specifically for a user defined function. For example if the user wants to find the root of x-a*sin(x)-b=0, for some parameters a and b, then Tootie Rootie might return the "update function" g(x) := x-sin(f(x)) and the initial guess x_0 := b. Actually, it will output Python code in the form of a string, which easily can be pluged into the user's existing code base.

# Restrictions and assumptions
 - Only Python 3.x support.
 - Only functions with 1D output are supported.
