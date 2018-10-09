from sympy import *
from sympy.solvers.solveset import linsolve

# zfc, zsb, zsc, phi, cons, t, f, d, gamma, St, st, con, b, pe, R, pt, a, V, f, d, con = symbols('zfc, zsb, zsc, phi, cons, t, f, d, gamma, St, st, con, b, pe, R, pt, a, V, f, d, con')
#
# params = linsolve([phi * (cons * t - zsc - f - zfc - d) + gamma * (St + zsb + st - zfc - con - b/2 - 5 * pe),
#  pe - R * pt - 2 * a * (zsb + zsc) * V + gamma * (St + zsb + st - zfc - con - b / 2 - 5 * pe),
#   pe - R * pt - 2 * a * (zsb + zsc) * V + phi * (cons * t - zsc - f - zfc - d)], (zfc, zsb, zsc))

a1, a2, a3, phi, a, V, gamma, zsc, zsb, zfc = symbols('a1, a2, a3, phi, a, V, gamma, zsc, zsb, zfc')

params = linsolve([a1 - phi * zsc + gamma * zsb - zfc * (gamma + phi), a2 - zsb * ( 2 * a * V + gamma) - 2 * a * V * zsc + gamma * zfc, a3 - 2 * a * V * zsb - zsc * ( 2 * a * V + phi) - phi * zfc], (zfc, zsb, zsc))
