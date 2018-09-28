import numpy as np
import scipy.optimize as sc
import matplotlib.pyplot as plt

#demand curve on an aggergate level
def DemandCurve(time, c, r, prices, stored, belief, bCap, aversion, volatility, consumption, n, spike):
    #get the forecasts first
    push = 5
    gamma = 0.001
    forecastPrice1 = GetForecast(c, r, prices, 1,  aversion, n, volatility, time, consumption, stored, bCap, spike)
    forecastPrice2 = GetForecast(c, r, prices, 2,  aversion, n, volatility, time, consumption, stored, bCap, spike)

    #find the individual demands for different forecasts
    forecast = belief * forecastPrice1 + (1 - belief) * forecastPrice2

    aggregateDemand = consumption + c + (gamma * (bCap / 2 - stored) + forecast * (gamma * push + 1) - (1 + r) * prices[-1]) / (2 * aversion * volatility + gamma) #((bCap / 2 - stored - c + push * forecast) + 1000 *(forecast - (1 + r) * prices[-1])) / (2 * aversion * volatility + 1)
    # print(aggregateDemand)
    return aggregateDemand

#generation function
def GetForecast(c, r, prices, belief, aversion, n, volatility, time, consumption, stored, bCap, spike):
    push = 5
    gamma = 0.001
    fundamentalPrice = ((2 * aversion * volatility + gamma) * (c - Generation(consumption, c, time, spike) + consumption) + gamma * (bCap / 2 - stored)) / (n * r - gamma * push)

    if (belief == 1):
        return fundamentalPrice
    else:
        return fundamentalPrice + 1.01 * (prices[-2] - fundamentalPrice)

#generation function
def Generation(consumption, c, time, spike):
    if (time == 70): return consumption + c + spike
    if (time == 80): return consumption + c - spike
    return consumption + c#+ int(np.random.uniform(0, 1) < 0.02) * np.random.uniform(-10,10)
# + 2 * np.random.uniform(-1,1) +
#market clearing
def MarketClearing(time, c, r, prices, stored, belief, bCap, aversion, volatility, consumption, n, spike):
    f =  lambda p: DemandCurve(time, c, r, np.append(prices, p), stored, belief, bCap, aversion, volatility, consumption, n, spike) - Generation(consumption, c, time, spike)
    price = sc.bisect(f, -10**10, 10**10)
    deltaStore = DemandCurve(time, c, r, np.append(prices, price), stored, belief, bCap, aversion, volatility, consumption, n, spike) - consumption

    return price, deltaStore

#percentage of opinion 1
def ForecastPercentage(beta, c, r, prices, aversion, cost, n, time, consumption, belief, eta, stored, bCap, spike):
    pi1 = - ( GetForecast(c, r, prices, 1, aversion, n, volatility, time, consumption, stored, bCap, spike) - prices[-1]) ** 2 - cost
    pi2 = - ( GetForecast(c, r, prices, 2, aversion, n, volatility, time, consumption, stored, bCap, spike) - prices[-1]) ** 2

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

    beta = 0.03
    eta = 0.1
    # cost = 10
    beliefAvg = []
    storage = []
    variances = []
    derivatives = []
    c = 2 * np.sin(np.linspace(1, time/12, time))
    distrubance = 0.01
    spike = 20

    #loop over time
    for cost in np.arange(0, 50, 0.5):
        store = np.array([bCap / 2])
        for i in range(time):
            newPrice, delta = MarketClearing(i, c[i], r, prices, store[-1], belief[-1], bCap, aversion, volatility, consumption, n, spike)
            prices2[-1] = prices[-1] + distrubance
            newPrice2, delta2 = MarketClearing(i, c[i], r, prices2, store[-1], belief[-1], bCap, aversion, volatility, consumption, n, spike)

            prices = np.append(prices, newPrice)
            prices2 = np.append(prices2, newPrice2)
            store = np.append(store, store[-1] + delta)
            belief = np.append(belief, ForecastPercentage(beta, c[i], r, prices, aversion, cost, n, time, consumption, belief, eta, store[-1], bCap, spike))
            variances = np.append(variances, np.var(prices))

        beliefAvg = np.append(beliefAvg, np.mean(belief[-time:]))
        variances = np.append(variances, np.var(prices[-time:]))
        storage = np.append(storage, store[-time:])
        derivative = np.log(abs((prices2[-time:] - prices[-time:])/distrubance))
        # derivative = derivative[derivative >= -1E308]
        derivatives = np.append(derivatives, np.mean(derivative))
    # plt.plot(derivatives)
    # # plt.yscale('log')
    # plt.xlabel('Spike Size')
    # plt.ylabel('LPE')
    # plt.tight_layout()
    # plt.show()
