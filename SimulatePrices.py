import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as sk
import scipy.optimize as opt

def Simulation(starting, alpha, sigma, time, eta, cutoff, size):
    x = np.zeros([time, 1])
    x[0] = starting
    # generate the simulation
    for i in range(1, time):
        dx = -alpha * x[i] + eta * (sigma * np.random.normal(0, 1)) + (1 - eta) * (x[i - 1])
        # if(i != 1): x[i - 1] = x[i - 1] + int(cutoff > np.random.uniform(0, 1)) * np.random.uniform(-size, size)
        x[i] = x[i] + dx

    # x[time - 1] = x[time - 1] + int(cutoff > np.random.uniform(0, 1)) * np.random.uniform(-size, size)

    return x

if(__name__ == '__main__'):
    bCap = 100
    numSim = 1000
    time = 10
    r = 0.05 / time
    startVol = 50
    alpha = 10
    sigma = 3
    eta = 0.3
    cutoff = 0.005
    size = 100
    # starting = -1
    simulatedPrices = np.zeros([numSim, 10])

    for starting in range(-500, 3001, 50):
        for i in range(0, numSim):
            simulatedPrices[i] = Simulation(starting, alpha, sigma, time, eta, cutoff, size).reshape([time])
            np.savetxt("prices" + str(starting) +".csv", simulatedPrices, delimiter=",")
