import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as sk
import scipy.optimize as opt

# simulations
def Simulation(alpha, sigma, time, eta, cutoff, size):
    x = np.zeros([time, 1])
    x[0] = 0.01
    # generate the simulation
    for i in range(1, time - 1):
        dx = -alpha * x[i] + eta * (sigma * np.sqrt(abs(x[i])) * np.random.normal(0, 1)) + (1 - eta) * (sigma * np.sqrt(abs(x[i - 1])))
        if(i != 1): x[i - 1] = x[i - 1] + int(cutoff > np.random.uniform(0, 1)) * np.random.uniform(-size, size)
        x[i + 1] = x[i] + dx

    x[time - 1] = x[time - 1] + int(cutoff > np.random.uniform(0, 1)) * np.random.uniform(-size, size)

    return x

# function to be optimized
def Function(dv, price, storage, bCap, type, lm, time, i):
    if (type == 1):
        if(storage - dv > bCap):
            return -10**99
        elif(storage - dv < 0):
            return -10**99
        else:
            return -price * dv
    elif (type == 2):
        return startVol - storage
    else:
        if(storage - dv > bCap):
            return -10**99
        elif(storage - dv < 0):
            return -10**99
        else:
            return -price * dv + np.exp(r) *( lm.coef_[0] * (time - i) + lm.coef_[1] * (time - i)**2 + lm.coef_[2] * (time - i)**3 + lm.coef_[3] * prices + lm.coef_[4] * prices**2 + lm.coef_[5] * prices**3 + lm.coef_[6] * storage + lm.coef_[7] * storage**2 + lm.coef_[8] * storage**3)

# find the optimal value
def Optimize(prices, storage, lm, type, numSim, bCap, time, i, r):
        solutions = np.zeros([numSim])
        dv = np.zeros([numSim])
        for j in range(numSim):
            res = opt.maximize(Function, 0.01, args = (prices[j], storage[j], bCap, type, lm, time, i, r),  method = 'nelder-mead')
            solutions[j] = res.fun
            dv[j] = res.x
        return solutions, dv

# OPTIMIZATION
def BackwardInduction(prices, time, storage, numSim, bCap, continuation, r):
    # use the coefficients to find the objective function, and optimize it
    # first result
    # WARNING: CHECK IF ALL THE INPUTS ARE CORRECT!!
    continuation[-2], dv = Optimize(prices[:,-1], storage[:,-1], _, 1, numSim, bCap, time, time - 1, r)
    storage[-2] = storage[-1] - dv

    # middle results
    for i in range(2, time - 1):
        # regress and find the continuation function
        model = sk.LinearRegression(intercept=True)
        lm = model.fit(np.array([(time - i), (time - i)**2, (time - i)**3, prices[:,-i], prices[:,-i]**2, prices[:,-i]**3, storage[:, -i], storage[:, -i]**2, storage[:, -i]**3]).T, continuation[-2][i])
        continuation[-i], dv = Optimize(prices[:,-i], storage[:,-i], _, 1, numSim, bCap, time, i, r)
        storage[-i] = storage[-i + 1] - dv

    # last result
    i = time
    lm = model.fit(np.array([(time - i), (time - i)**2, (time - i)**3, prices[:,-i], prices[:,-i]**2, prices[:,-i]**3, storage[:, -i], storage[:, -i]**2, storage[:, -i]**3]).T, continuation[0][i])
    continuation[0], dv = Optimize(prices[:,0], storage[:,0], lm, 2, numSim, bCap, time, 0, r)
    storage[0] = storage[1] - dv

    return continuation

if(__name__ == '__main__'):
    bCap = 100
    numSim = 1000
    time = 1000
    r = 0.05 / time
    prices = np.zeros([numSim, time])
    storage = np.zeros([numSim, time])
    storage[:, -1] = bCap / 2 * np.ones([numSim])
    continuation = np.zeros([numSim, time])

    for i in range(numSim):
        sim = Simulation(0.2, 0.2, time, 0.9, 0.05, 100)
        prices[i] = sim.reshape([time])

    continuation = BackwardInduction(prices, time, storage, numSim, bCap, continuation, r)



# TODO:
# 1) OPTIMIZATION
# a) find the continuation
# 2) REGRESSION TO GET CONTINUOATION function
# 3)
