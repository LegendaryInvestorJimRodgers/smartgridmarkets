import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def prices(gen, cons, fut, n, n1, r, V, b, a):
    fund = (2 * a * V * (fut - gen + cons)) / (n * r)
    return ((gen - cons - fut) * 2 * a * V - n1 * fund - (1 - n1) * fund + (1 - n1) * b * fund) / ((1 - n1) * b - (1 + r))

def opinions(n, a, V, fut, gen, cons, r, cost, eta, beta, price, beliefs):
    fund = (2 * a * V * (fut - gen + cons)) / (n * r)
    pi1 = - (fund - price[-1]) ** 2 - cost
    pi2 = - (fund + b * (price[-1] - fund))
    return eta * 1 / (1 + np.exp(beta * pi2 - beta * pi1) ) + (1 - eta) * beliefs[-1]

if (__name__ == '__main__'):
    fut = 0
    cons = 40.1
    gen = 40
    a = 10
    V = 3
    n1 = 0.5
    n = 10
    r = 0.05
    b = 1.12
    beta = 0.5
    eta = 0.5
    cost = 0
    time = 1000

    startP = -50
    endP = 50
    tickP = 30
    startO = 0
    endO = 1
    tickO = 30


    # first comes the beliefs calculation then the prices
    # for i in range(time):
    #     beliefs = np.append(beliefs, opinions(c, r, price, cost, b, beliefs, eta))
    #     price = np.append(price, prices(gen, cons, a, V, n, c, r, b))

    # dP = [price[i + 1] - price[i] for i in range(len(price) - 1)]
    # dB = [beliefs[i + 1] - beliefs[i] for i in range(len(beliefs) - 1)]
    def f(Y, t):
        y1, y2 = Y
        price = np.array([y1])
        beliefs = np.array([y2])
        beliefNext = opinions(n, a, V, fut, gen, cons, r, cost, eta, beta, price, beliefs)
        priceNext = prices(gen, cons, fut, n, n1, r, V, b, a)
        return [(priceNext - price[-1])/((endP - startP) / tickP), (beliefNext - beliefs[-1]) / ((endO - startO) / tickO) ]

    y1 = np.linspace(startP, endP, tickP)
    y2 = np.linspace(startO, endO, tickO)

    Y1, Y2 = np.meshgrid(y1, y2)

    t = 0

    u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)

    NI, NJ = Y1.shape

    for i in range(NI):
        for j in range(NJ):
            x = Y1[i, j]
            y = Y2[i, j]
            yprime = f([x, y], t)
            u[i, j] = yprime[0]
            v[i, j] = yprime[1]


    Q = plt.quiver(Y1, Y2, u, v, color='r')

    for y20 in [ 0.2, 0.5, 0.9]:
        tspan = np.linspace(0, 50, 200)
        y0 = [5, y20]
        ys = odeint(f, y0, tspan)
        plt.plot(ys[:, 0], ys[:, 1], 'b-')  # path
        plt.plot([ys[0, 0]], [ys[0, 1]], 'o')  # start
        plt.plot([ys[-1, 0]], [ys[-1, 1]], 's')  # end

    plt.xlim([-startP, endP])
    plt.xlabel('Price')
    plt.ylabel('Belief')
    plt.xlim([startP, endP])
    plt.ylim([startO, endO])
    plt.show()
