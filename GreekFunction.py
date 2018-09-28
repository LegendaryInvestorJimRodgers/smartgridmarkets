import numpy as np
import sklearn.linear_model as sk
from numpy import genfromtxt

# derivative
prices = genfromtxt('batteryprices7.csv', delimiter=",")
prices = prices[1:]
prices = np.diff(prices, n=1)

model = sk.LinearRegression(fit_intercept=True)
startingPrices = np.sort(np.tile(range(-100, 101, 50), 20))
startingVolume = np.tile(range(0, 101, 5), 5)

lm = model.fit(np.array([startingPrices]).T, prices.reshape(1420, 1))
print(lm.score(np.array([startingPrices]).T, prices.reshape(1420, 1)))
print('coefficient: ', lm.coef_, 'intercept: ', lm.intercept_)

# pay-off
data = genfromtxt('batteryprices6.csv', delimiter=",")
data = data[1:]
startingPrices = np.sort(np.tile(range(-500, 3001, 50), 21))
startingVolume = np.tile(range(0, 101, 5), 71)

lm2 = model.fit(np.array([startingPrices, startingPrices * startingVolume]).T, data.reshape(1491, 1))
print(lm.score(np.array([startingPrices, startingPrices * startingVolume]).T, data.reshape(1491, 1)))
print('coefficient: ', lm.coef_, 'intercept: ', lm.intercept_)
