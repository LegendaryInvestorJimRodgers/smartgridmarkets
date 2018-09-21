import numpy as np
from numpy import genfromtxt
import scipy.optimize as opt
import sklearn.linear_model as sk
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def Function(dv, price, price2, storage, bCap, type, lm, time, i, r, startVol):
    if (type == 1):
        if(storage - dv > bCap):
            return 10**99
        elif(storage - dv < 0):
            return 10**99
        else:
            return price * dv
    elif (type == 2 ):
        return -(startVol - storage) * price - np.exp(r) * ( lm.intercept_ + lm.coef_[0] * price2 + lm.coef_[1] * (startVol - storage) )
    else:
        if(storage - dv > bCap):
            return 10**99
        elif(storage - dv < 0):
            return 10**99
        else:
            return -1 * (-price * dv + np.exp(-r) * ( lm.intercept_ + lm.coef_[0] * price2 + lm.coef_[1] * (storage - dv) ))

# find the optimal value
def Optimize(prices, prices2, storage, lm, type, numSim, bCap, time, i, r, startVol):
        solutions = np.zeros([numSim])
        dv = np.zeros([numSim])
        for j in range(numSim):
            res = opt.minimize(Function, 0, args = (prices[j], prices[j], storage[j], bCap, type, lm, time, i, r, startVol),  method = 'nelder-mead')
            solutions[j] = res.fun
            dv[j] = res.x
        return -solutions, dv

length = 1000
bCap = 100
startVol = 0
r = 0.02 / 10
# batteryPrices = np.zeros([100, 2])
counter = 0
i = 1
startVol = 50
endVol = bCap / 2
batteryMat = np.zeros([21])

for i in range(-500, 3001, 50):
    prices = genfromtxt('prices' + str(i) + '.csv', delimiter=",")
    temp = []

    for startVol in range(0, bCap + 1, 5):
        # first part
        solution, dv = Optimize(prices[:, -1], "NA", np.ones([length, 1]) * endVol, "NA", 1, length, bCap, 10, 10, r, startVol)
        s = np.ones([length, 1]).T * endVol
        storage = np.ones([length, 1]).T * endVol - dv

        model = sk.Ridge(fit_intercept=True, alpha = 0.1)
        lm = model.fit(np.array([prices[:,-1].reshape([length]), np.ones([length, 1]).reshape([length]) * endVol]).T, np.zeros([length,1]).reshape([length]))
        lm2 = model.fit(np.array([prices[:,-1].reshape([length]), np.ones([length, 1]).reshape([length]) * endVol]).T, solution.reshape([length]))

        solution2, dv2 = Optimize(prices[:, -2], prices[:, -1], storage[0], lm2, 0, length, bCap, 10, 9, r, startVol)
        s2 = storage
        storage = storage[0] - dv2
        lm3 = model.fit(np.array([prices[:,-2].reshape([length]), storage.reshape([length])]).T, solution2.reshape([length]))

        solution3, dv3 = Optimize(prices[:, -3], prices[:, -2], storage, lm3, 0, length, bCap, 10, 8, r, startVol)
        s3 = storage
        storage = (storage - dv3)
        lm4 = model.fit(np.array([prices[:,-3].reshape([length]), storage.reshape([length])]).T, solution3.reshape([length]))

        solution4, dv4 = Optimize(prices[:, -4], prices[:, -3], storage, lm4, 0, length, bCap, 10, 7, r, startVol)
        s4 = storage
        storage = (storage - dv4)
        lm5 = model.fit(np.array([prices[:,-4].reshape([length]), storage.reshape([length])]).T, solution4.reshape([length]))

        solution5, dv5 = Optimize(prices[:, -5], prices[:, -4], storage, lm5, 0, length, bCap, 10, 6, r, startVol)
        s5 = storage
        storage = (storage - dv5)
        lm6 = model.fit(np.array([prices[:,-5].reshape([length]), storage.reshape([length])]).T, solution5.reshape([length]))

        solution6, dv6 = Optimize(prices[:, -6], prices[:, -5], storage, lm6, 0, length, bCap, 10, 5, r, startVol)
        s6 = storage
        storage = (storage - dv6)
        lm7 = model.fit(np.array([prices[:,-6].reshape([length]), storage.reshape([length])]).T, solution6.reshape([length]))

        solution7, dv7 = Optimize(prices[:, -7], prices[:, -6], storage, lm7, 0, length, bCap, 10, 4, r, startVol)
        s7 = storage
        storage = (storage - dv7)
        lm8 = model.fit(np.array([prices[:,-7].reshape([length]), storage.reshape([length])]).T, solution7.reshape([length]))

        solution8, dv8 = Optimize(prices[:, -8], prices[:, -7], storage, lm8, 0, length, bCap, 10, 3, r, startVol)
        s8 = storage
        storage = (storage - dv8)
        lm9 = model.fit(np.array([prices[:,-8].reshape([length]), storage.reshape([length])]).T, solution8.reshape([length]))

        solution9, dv9 = Optimize(prices[:, -9], prices[:, -8], storage, lm9, 0, length, bCap, 10, 2, r, startVol)
        s9 = storage
        storage = (storage - dv9)
        lm10 = model.fit(np.array([prices[:,-9].reshape([length]), storage.reshape([length])]).T, solution9.reshape([length]))

        solution10, _ = Optimize(prices[:, -10], prices[:, -9], storage, lm10, 2, length, bCap, 10, 1, r, startVol)
        dv10 = startVol - storage

        finalPrice = np.mean(solution10)
        # batteryPrices[counter] = np.array([finalPrice, startVol])

        # # second part
        # delta = 0.0001
        # startVol = startVol + delta
        #
        # solution, dv = Optimize(prices[:, -1], "NA", np.ones([length, 1]) * endVol, "NA", 1, length, bCap, 10, 10, r, startVol)
        # s = np.ones([length, 1]).T * endVol
        # storage = np.ones([length, 1]).T * endVol - dv
        #
        # model = sk.Ridge(fit_intercept=True, alpha = 0.1)
        # lm = model.fit(np.array([prices[:,-1].reshape([length]), np.ones([length, 1]).reshape([length]) * endVol]).T, np.zeros([length,1]).reshape([length]))
        # lm2 = model.fit(np.array([prices[:,-1].reshape([length]), np.ones([length, 1]).reshape([length]) * endVol]).T, solution.reshape([length]))
        #
        # solution2, dv2 = Optimize(prices[:, -2], prices[:, -1], storage[0], lm2, 0, length, bCap, 10, 9, r, startVol)
        # s2 = storage
        # storage = storage[0] - dv2
        # lm3 = model.fit(np.array([prices[:,-2].reshape([length]), storage.reshape([length])]).T, solution2.reshape([length]))
        #
        # solution3, dv3 = Optimize(prices[:, -3], prices[:, -2], storage, lm3, 0, length, bCap, 10, 8, r, startVol)
        # s3 = storage
        # storage = (storage - dv3)
        # lm4 = model.fit(np.array([prices[:,-3].reshape([length]), storage.reshape([length])]).T, solution3.reshape([length]))
        #
        # solution4, dv4 = Optimize(prices[:, -4], prices[:, -3], storage, lm4, 0, length, bCap, 10, 7, r, startVol)
        # s4 = storage
        # storage = (storage - dv4)
        # lm5 = model.fit(np.array([prices[:,-4].reshape([length]), storage.reshape([length])]).T, solution4.reshape([length]))
        #
        # solution5, dv5 = Optimize(prices[:, -5], prices[:, -4], storage, lm5, 0, length, bCap, 10, 6, r, startVol)
        # s5 = storage
        # storage = (storage - dv5)
        # lm6 = model.fit(np.array([prices[:,-5].reshape([length]), storage.reshape([length])]).T, solution5.reshape([length]))
        #
        # solution6, dv6 = Optimize(prices[:, -6], prices[:, -5], storage, lm6, 0, length, bCap, 10, 5, r, startVol)
        # s6 = storage
        # storage = (storage - dv6)
        # lm7 = model.fit(np.array([prices[:,-6].reshape([length]), storage.reshape([length])]).T, solution6.reshape([length]))
        #
        # solution7, dv7 = Optimize(prices[:, -7], prices[:, -6], storage, lm7, 0, length, bCap, 10, 4, r, startVol)
        # s7 = storage
        # storage = (storage - dv7)
        # lm8 = model.fit(np.array([prices[:,-7].reshape([length]), storage.reshape([length])]).T, solution7.reshape([length]))
        #
        # solution8, dv8 = Optimize(prices[:, -8], prices[:, -7], storage, lm8, 0, length, bCap, 10, 3, r, startVol)
        # s8 = storage
        # storage = (storage - dv8)
        # lm9 = model.fit(np.array([prices[:,-8].reshape([length]), storage.reshape([length])]).T, solution8.reshape([length]))
        #
        # solution9, dv9 = Optimize(prices[:, -9], prices[:, -8], storage, lm9, 0, length, bCap, 10, 2, r, startVol)
        # s9 = storage
        # storage = (storage - dv9)
        # lm10 = model.fit(np.array([prices[:,-9].reshape([length]), storage.reshape([length])]).T, solution9.reshape([length]))
        #
        # solution10, _ = Optimize(prices[:, -10], prices[:, -9], storage, lm10, 2, length, bCap, 10, 1, r, startVol)
        # dv10 = startVol - storage
        #
        # finalPrice2 = np.mean(solution10)
        temp = np.append(temp, finalPrice)


        counter = counter + 1
    print(i)
    batteryMat = np.row_stack((batteryMat, temp))

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# x = batteryPrices[:, 2]
# y = batteryPrices[:, 1]
# z = batteryPrices[:, 0]
# ax.scatter(x, y, z, alpha = 0.8)
# plt.show()
