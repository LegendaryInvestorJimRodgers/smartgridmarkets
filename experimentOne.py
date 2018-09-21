import numpy as np
import scipy.optimize as sc
import matplotlib.pyplot as plt

#demand curve on an aggergate level
def DemandCurve(time, c, r, prices, stored, belief, bCap, aversion, volatility, consumption, n):
    #get the forecasts first
    push = 5
    forecastPrice1 = GetForecast(c, r, prices, 1,  aversion, n, volatility, time, consumption, stored, bCap)
    forecastPrice2 = GetForecast(c, r, prices, 2,  aversion, n, volatility, time, consumption, stored, bCap)

    #find the individual demands for different forecasts
    forecast = belief * forecastPrice1 + (1 - belief) * forecastPrice2

    aggregateDemand = consumption + c + ((bCap / 2 - stored - c + push * forecast) + 700 * (forecast - (1 + r) * prices[-1])) / (2 * aversion * volatility + 1)
    # print(aggregateDemand)
    return aggregateDemand

#generation function
def GetForecast(c, r, prices, belief, aversion, n, volatility, time, consumption, stored, bCap):
    push = 5
    fundamentalPrice = ((2 * aversion * volatility + 1) * (c - Generation(consumption, c, time) + consumption) ) / (n * (1 + r + 1 + push)) + ( bCap / 2 - stored) / (1 + r + 1 + push)

    if (belief == 1):
        return fundamentalPrice
    else:
        return fundamentalPrice + 1.01 * (prices[-2] - fundamentalPrice)

#generation function
def Generation(consumption, c, time):
    # if (time == 700): return consumption + 30
    # if (time == 800): return consumption - 200
    # return consumption
    return consumption + int(np.random.uniform(0, 1) < 0.02) * np.random.uniform(-10,10)
# + 2 * np.random.uniform(-1,1) +
#market clearing
def MarketClearing(time, c, r, prices, stored, belief, bCap, aversion, volatility, consumption, n):
    f =  lambda p: DemandCurve(time, c, r, np.append(prices, p), stored, belief, bCap, aversion, volatility, consumption, n) - Generation(consumption, c, time)
    price = sc.bisect(f, -10**10, 10**10)
    deltaStore = DemandCurve(time, c, r, np.append(prices, price), stored, belief, bCap, aversion, volatility, consumption, n) - consumption

    return price, deltaStore

#percentage of opinion 1
def ForecastPercentage(beta, c, r, prices, aversion, cost, n, time, consumption, belief, eta, stored, bCap):
    pi1 = - ( GetForecast(c, r, prices, 1, aversion, n, volatility, time, consumption, stored, bCap) - prices[-1]) ** 2 - cost
    pi2 = - ( GetForecast(c, r, prices, 2, aversion, n, volatility, time, consumption, stored, bCap) - prices[-1]) ** 2

    return eta * 1 / (1 + np.exp(beta * (pi2 - pi1))) + (1 - eta) * belief[-1]

#characteristics for aggregate
if (__name__ == '__main__'):
    r = 0.02
    prices = np.array([0])
    belief = np.array([0.5])
    bCap = 100
    aversion = 2
    volatility = 3
    consumption = 40
    time = 5000
    n = 1000
    store = np.array([bCap / 2])
    beta = 3
    eta = 0.5
    cost = 0.1
    variances = []
    c = 2 * np.sin(np.linspace(1, time/12, time))

    #loop over time
    # for cost in np.arange(0.0025, 0.035, 0.002):
    for i in range(time):
        newPrice, delta = MarketClearing(i, c[i], r, prices, store[-1], belief[-1], bCap, aversion, volatility, consumption, n)

        prices = np.append(prices, newPrice)
        store = np.append(store, store[-1] + delta)
        belief = np.append(belief, ForecastPercentage(beta, c[i], r, prices, aversion, cost, n, time, consumption, belief, eta, store[-1], bCap))
        variances = np.append(variances, np.var(prices))
