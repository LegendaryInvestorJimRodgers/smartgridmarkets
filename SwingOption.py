import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as sk
import scipy.optimize as opt

# simulations
def Simulation(alpha, sigma, time, eta, cutoff, size):
    x = np.zeros([time, 1])
    # generate the simulation
    for i in range(1, time):
        dx = -alpha * x[i] + eta * (sigma * np.random.normal(0, 1)) + (1 - eta) * (sigma * abs(x[i - 1]))
        # if(i != 1): x[i - 1] = x[i - 1] + int(cutoff > np.random.uniform(0, 1)) * np.random.uniform(-size, size)
        x[i] = x[i] + dx

    # x[time - 1] = x[time - 1] + int(cutoff > np.random.uniform(0, 1)) * np.random.uniform(-size, size)

    return x

# function to be optimized
def Function(dv, price, price2, storage, bCap, type, lm, time, i, r, startVol):
    if (type == 1):
        if(storage - dv > bCap):
            return 10**99
        elif(storage - dv < 0):
            return 10**99
        else:
            return price * dv
    elif (type == 2):
        return (startVol - storage) * price + np.exp(r) * ( lm.intercept_ + lm.coef_[0] * price2 + lm.coef_[1] * (startVol - storage) )
    else:
        if(storage - dv > bCap):
            return 10**99
        elif(storage - dv < 0):
            return 10**99
        else:
            return -1 * (-price * dv + np.exp(r) * ( lm.intercept_ + lm.coef_[0] * price2 + lm.coef_[1] * (storage - dv) ))

# find the optimal value
def Optimize(prices, prices2, storage, lm, type, numSim, bCap, time, i, r, startVol):
        solutions = np.zeros([numSim])
        dv = np.zeros([numSim])
        for j in range(numSim):
            res = opt.minimize(Function, 0, args = (prices[j], prices[j], storage[j], bCap, type, lm, time, i, r, startVol),  method = 'nelder-mead')
            solutions[j] = res.fun
            dv[j] = res.x
        return -solutions, dv

# OPTIMIZATION
def BackwardInduction(prices, time, storage, numSim, bCap, continuation, r, startVol):
    # use the coefficients to find the objective function, and optimize it
    # first result
    continuation[:, -2], dv = Optimize(prices[:,-1], 'NA', storage[:,-1], 'NA', 1, numSim, bCap, time, time - 1, r, startVol)
    storage[:, -2] = storage[:, -1] - dv
    length = len(prices[:, 0])
    model = sk.LinearRegression(fit_intercept=True)

    # middle results
    for i in range(3, time - 1):
        # regress and find the continuation
        lm = model.fit(np.array([prices[:,-i + 1], storage[:,-i + 1]]).T, continuation[:, -i + 1])
        continuation[:, -i - 1], dv = Optimize(prices[:,-i], prices[:, -i + 1], storage[:,-i + 1], lm, 0, numSim, bCap, time, i, r, startVol)
        storage[:, -i - 1] = storage[:, -i] - dv
        print(lm.intercept_, lm.coef_[0], lm.coef_[1])

    # last result
    i = time
    lm = model.fit(np.array([prices[:,-i + 1], storage[:,-i + 1]]).T, continuation[:, -i + 1])
    continuation[:, 0], dv = Optimize(prices[:,1], prices[:,2], storage[:,1], lm, 2, numSim, bCap, time, 0, r, startVol)
    storage[:, 0] = startVol
    return continuation

if(__name__ == '__main__'):
    bCap = 100
    numSim = 100
    time = 10
    r = 0.05 / time
    startVol = 50
    prices = np.zeros([numSim, time])
    storage = np.zeros([numSim, time])
    storage[:, -1] = bCap / 2 * np.ones([numSim])
    continuation = np.zeros([numSim, time])

    for i in range(numSim):
        sim = Simulation(10, 10, time, 0.9, 0.0005, 100)
        prices[i] = sim.reshape([time])

    continuation = BackwardInduction(prices, time, storage, numSim, bCap, continuation, r, startVol)
    price = np.mean(continuation.sum(axis=1))
