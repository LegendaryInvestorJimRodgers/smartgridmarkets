import numpy as np
from sympy import *
import sympy

cost, beta, b, pi1, pi2, r, n, a, V, p, n1, fund, b, gen, cons, fut = symbols('cost, beta, b, pi1, pi2, r, n, a, V, p, n1, fund, b, gen, cons, fut')
fund = (2 * a * V * (fut - gen + cons)) / (n * r)

# prices
fp = (gen - cons - fut) * 2 * a * V - n1 * fund  - (1 - n1) * fund + (1 - n1) * b * fund / ((1 - n1) * b - (1 + r))

# beliefs
pi1 = - ( fund - p ) ** 2 - cost
pi2 = - ( fund + b * (p - fund) - p ) ** 2

fn = 1 / (1 + exp(beta * pi2 - beta * pi1) )

# jacobian
F = sympy.Matrix([fp, fn])
J = F.jacobian([p, n1])

# eigenvalues
# eigs = J.eigenvals()

# find the places where eigs are less tha one
