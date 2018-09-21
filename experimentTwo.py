import numpy as np
import scipy.optimize as sc
import matplotlib.pyplot as plt

#demand curve on an aggergate level
def DemandCurve(c, r, prices, stored, belief, bCap, aversion, volatility, consumption, n):
    #get the forecasts first
    forecastPrice1 = GetForecast(c, r, prices, 1,  aversion, n, volatility)
    forecastPrice2 = GetForecast(c, r, prices, 2,  aversion, n, volatility)

    #find the individual demands for different forecasts
    forecast = belief * forecastPrice1 + (1 - belief) * forecastPrice2

    aggregateDemand = consumption + c + ( forecast - (1 + r) * prices[-1]) / (2 * aversion * volatility)

    #check the constraints for charging
    if(aggregateDemand + stored - consumption > bCap):
        aggergateDemand = consumption + c + bCap - stored
    elif(aggregateDemand + stored - consumption < 0):
        aggregateDemand = consumption + c - stored

    return aggregateDemand

#generation function
def GetForecast(c, r, prices, belief, aversion, n, volatility):
    fundamentalPrice = (2 * aversion * volatility * c) / (n * r)

    if (belief == 1):
        return fundamentalPrice
    else:
        return fundamentalPrice - 1.01 * (fundamentalPrice - prices[-1])

#generation function
def Generation(consumption, time):
    # if (time == 700): return consumption + 20
    # if (time == 800): return consumption - 20
    # else
    return consumption + c[time]

#market clearing
def MarketClearing(time, c, r, prices, stored, belief, bCap, aversion, volatility, consumption, n):
    f =  lambda p: DemandCurve(c, r, np.append(prices, p), stored, belief, bCap, aversion, volatility, consumption, n) - Generation(consumption, time)
    price = sc.bisect(f, -10**10, 10**10)
    deltaStore = DemandCurve(c, r, np.append(prices, price), stored, belief, bCap, aversion, volatility, consumption, n) - consumption

    return price, deltaStore

#percentage of opinion 1
def ForecastPercentage(beta, c, r, prices, aversion, cost, n):
    pi1 = - ( GetForecast(c, r, prices, 1, aversion, n, volatility ) - prices[-1]) ** 2 - cost
    pi2 = - ( GetForecast(c, r, prices, 2, aversion, n, volatility ) - prices[-1]) ** 2

    return 1 / (1 + np.exp(beta * (pi2 - pi1)))

#characteristics for aggregate
if (__name__ == '__main__'):
    r = 0.05
    prices = np.array([0, 0])
    belief = np.array([0.5])
    bCap = 100
    aversion = 2
    volatility = 3
    consumption = 40
    time = 1000
    n = 100000
    store = np.array([bCap / 2])
    beta = 2
    cost = 0
    variances = []
    c = 2 * np.sin(np.linspace(1, time/12, time))

    #loop over time
    # for cost in np.arange(0.0025, 0.035, 0.002):
    for i in range(time):
        newPrice, delta = MarketClearing(i, c[i], r, prices, store[-1], belief[-1], bCap, aversion, volatility, consumption, n)

        prices = np.append(prices, newPrice)
        store = np.append(store, store[-1] + delta)
        belief = np.append(belief, ForecastPercentage(beta, c[i], r, prices, aversion, cost, n))
        variances = np.append(variances, np.var(prices))
