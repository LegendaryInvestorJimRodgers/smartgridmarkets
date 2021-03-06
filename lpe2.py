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

    aggregateDemand = consumption + c +  ( forecast - (1 + r) * prices[-1]) / (2 * aversion * volatility)

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
        return fundamentalPrice + 1.03 * (prices[-2] - fundamentalPrice)

#generation function
def Generation(consumption, c, time):
    if (time == 70): return consumption + c + 20000
    if (time == 80): return consumption + c - 20000
    else: return consumption + c #* np.random.normal(1, .1)

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
    prices = np.array([0, 0])
    prices2 = prices
    belief = np.array([0.5])
    bCap = 50000
    aversion = 2
    volatility = 3
    consumption = 20000
    time = 100
    # n = 100

    beta = 0.03
    eta = 0.1
    cost = 2
    beliefAvg = []
    storage = []
    variances = []
    derivatives = []
    recoveryTime = []
    spikeSize = []
    c = 200 * np.sin(np.linspace(1, time/12, time))
    distrubance = 0.01

    #loop over time
    for n in np.arange(1, 100, 1):
        store = np.array([bCap / 2])
        for i in range(time):
            newPrice, delta = MarketClearing(i, c[i], r, prices, store[-1], belief[-1], bCap, aversion, volatility, consumption, n)
            prices2[-1] = prices[-1] + distrubance
            newPrice2, delta2 = MarketClearing(i, c[i], r, prices2, store[-1], belief[-1], bCap, aversion, volatility, consumption, n)

            prices = np.append(prices, newPrice)
            prices2 = np.append(prices2, newPrice2)
            store = np.append(store, store[-1] + delta)
            belief = np.append(belief, ForecastPercentage(beta, c[i], r, prices, aversion, cost, n, time, consumption, belief, eta))

        beliefAvg = np.append(beliefAvg, np.mean(belief[-time:]))
        variances = np.append(variances, np.var(prices[-time:]))
        storage = np.append(storage, store[-time:])
        derivative = np.log(abs((prices2[-time:] - prices[-time:])/distrubance))
        # derivative = derivative[derivative >= -1E308]
        derivatives = np.append(derivatives, np.mean(derivative))
        # recoveryTime = np.append(recoveryTime, np.where(abs(prices[(72 + time * cost):]) < 1e-5)[0][0])
        # spikeSize = np.append(spikeSize, np.max(abs(prices[(cost * time):])))
    print(beliefAvg)
    plt.plot(variances)
    # plt.yscale('log')
    plt.xlabel('cost')
    plt.ylabel('LPE')
    plt.tight_layout()
    plt.show()
    # plt.scatter(prices[1001:-1], belief[1000:-1])
    # plt.xlabel('prices')
    # plt.ylabel('beliefs')
    # plt.show()
