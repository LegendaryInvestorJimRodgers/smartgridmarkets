import numpy as np
import scipy.optimize as sc
import matplotlib.pyplot as plt

#demand curve on an aggergate level
def DemandCurve(time, c, r, prices, stored, belief, bCap, aversion, volatility, consumption, n):
    #get the forecasts first
    forecastPrice1 = GetForecast(c, r, prices, 1,  aversion, n, volatility, time, consumption)
    forecastPrice2 = GetForecast(c, r, prices, 2,  aversion, n, volatility, time, consumption)

    #find the individual demands for different forecasts
    forecast = belief * forecastPrice1 + (1 - belief) * forecastPrice2

    aggregateDemand = consumption + c + ( forecast - (1 + r) * prices[-1]) / (2 * aversion * volatility)

    #check the constraints for charging
    # if(aggregateDemand + stored - consumption > bCap):
    #     aggergateDemand = consumption - c + bCap - stored
    # elif(aggregateDemand - stored - consumption < 0):
    #     aggregateDemand = consumption - c - stored

    return aggregateDemand

#generation function
def GetForecast(c, r, prices, belief, aversion, n, volatility, time, consumption):
    fundamentalPrice = (2 * aversion * volatility * (c - Generation(consumption, c, time) + consumption)) / (n * r)

    if (belief == 1):
        return fundamentalPrice
    else:
        return fundamentalPrice + 1.01 * (prices[-2] - fundamentalPrice)

#generation function
def Generation(consumption, c, time):
    if (time == 70): return consumption + 20
    if (time == 80): return consumption - 20
    else: return consumption + c + 2 * np.random.uniform(-1,1)

#market clearing
def MarketClearing(time, c, r, prices, stored, belief, bCap, aversion, volatility, consumption, n):
    f =  lambda p: DemandCurve(time, c, r, np.append(prices, p), stored, belief, bCap, aversion, volatility, consumption, n) - Generation(consumption, c, time)
    price = sc.bisect(f, -10**10, 10**10)
    deltaStore = DemandCurve(time, c, r, np.append(prices, price), stored, belief, bCap, aversion, volatility, consumption, n) - consumption

    return price, deltaStore

#percentage of opinion 1
def ForecastPercentage(beta, c, r, prices, aversion, cost, n, time, consumption, belief, eta):
    pi1 = - ( GetForecast(c, r, prices, 1, aversion, n, volatility, time, consumption) - prices[-1]) ** 2 - cost
    pi2 = - ( GetForecast(c, r, prices, 2, aversion, n, volatility, time, consumption) - prices[-1]) ** 2

    return eta * 1 / (1 + np.exp(beta * (pi2 - pi1))) + (1 - eta) * belief[-1]

#characteristics for aggregate
if (__name__ == '__main__'):
    r = 0.05
    prices = np.array([2, -2])
    prices2 = prices
    belief = np.array([0.5])
    bCap = 100
    aversion = 2
    volatility = 3
    consumption = 40
    time = 100
    n = 100
    store = np.array([bCap / 2])
    beta = 0.03
    eta = 0.1
    # cost = 2
    variances = []
    derivatives = []
    c = 2 * np.sin(np.linspace(1, time/12, time))
    distrubance = 0.01

    #loop over time
    for cost in np.arange(0, 50, 0.5):
        for i in range(time):
            newPrice, delta = MarketClearing(i, c[i], r, prices, store[-1], belief[-1], bCap, aversion, volatility, consumption, n)
            prices2[-1] = prices[-1] + distrubance
            newPrice2, delta2 = MarketClearing(i, c[i], r, prices2, store[-1], belief[-1], bCap, aversion, volatility, consumption, n)

            prices = np.append(prices, newPrice)
            prices2 = np.append(prices2, newPrice2)
            store = np.append(store, store[-1] + delta)
            belief = np.append(belief, ForecastPercentage(beta, c[i], r, prices, aversion, cost, n, time, consumption, belief, eta))
            variances = np.append(variances, np.var(prices))

        derivative = np.log(abs((prices2 - prices)/distrubance))
        derivative = derivative[derivative >= -1E308]
        derivatives = np.append(derivatives, np.mean(derivative))
    plt.plot(-derivatives)
    plt.yscale('log')
    plt.xlabel('cost')
    plt.ylabel('LPE')
    plt.tight_layout()
    plt.show()
