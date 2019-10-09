import numpy as np
from sympy import lambdify, sympify
from scipy.integrate import odeint

var = ['x', 'y', 'z']
V = sympify("v**2/2*log(1+x**2 + (y/a)**2 + (z/c)**2)")
dVdvar_analytical = [V.diff(var[i]) for i in range(len(var))]
dVdvar = [lambdify(('x', 'y', 'z', 'v', 'a', 'c'), df) for df in dVdvar_analytical]

def afleiden(variables, _, params, dVdvar):
    x, y, z = variables
    v, a, c = params
    return [dVdvarj(x, y, z, v, a, c) for dVdvarj in dVdvar]

variables0, params = [0.3, 0.2, 0.9], [0.2, 0.6, 0.7]

t = np.arange(0, 10, .1)
y = odeint(afleiden, variables0, t, args=(params, dVdvar))
