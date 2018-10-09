import numpy as np
import scipy.optimize as sc
import matplotlib.pyplot as plt

#demand curve on an aggergate level
def DemandCurve(time, c, r, prices, stored,  belief1, belief2, belief3, bCap, aversion, volatility, consumption, n, spike):
    #get the forecasts first
    push = 5
    gamma = 0.001
    forecastPrice1 = GetForecast(c, r, prices, 1,  aversion, n, volatility, time, consumption, stored, bCap, spike)
    forecastPrice2 = GetForecast(c, r, prices, 2,  aversion, n, volatility, time, consumption, stored, bCap, spike)
    forecastPrice3 = GetForecast(c, r, prices, 3,  aversion, n, volatility, time, consumption, stored, bCap, spike)

    #find the individual demands for different forecasts
    forecast = belief1 * forecastPrice1 + belief2 * forecastPrice2 + belief3 * forecastPrice3

    aggregateDemand = consumption + c + (gamma * (bCap / 2 - stored - c + 20000) + forecast * (gamma * push + 1) - (1 + r) * prices[-1]) / (2 * aversion * volatility + gamma) #((bCap / 2 - stored - c + push * forecast) + 1000 *(forecast - (1 + r) * prices[-1])) / (2 * aversion * volatility + 1)
    # print(aggregateDemand)
    return aggregateDemand

#generation function
def GetForecast(c, r, prices, setting, aversion, n, volatility, time, consumption, stored, bCap, spike):
    push = 5
    gamma = 0.001
    fundamentalPrice = ((2 * aversion * volatility + gamma) * (c - Generation(consumption, c, time, spike) + consumption) + gamma * (bCap / 2 - stored + 20000 - c)) / (n * r + gamma * push)

    if (setting == 1):
        return fundamentalPrice
    elif(setting == 2):
        return fundamentalPrice + 1.03 * (prices[-2] - fundamentalPrice)
    elif(setting == 3):
        return fundamentalPrice - 20

#generation function
def Generation(consumption, c, time, spike):
    if (time == 70): return consumption + c# + spike
    if (time == 80): return consumption + c# - spike
    return consumption + c #* np.random.normal(0,1)
# + 2 * np.random.uniform(-1,1) +
#market clearing
def MarketClearing(time, c, r, prices, stored, belief1, belief2, belief3, bCap, aversion, volatility, consumption, n, spike):
    f =  lambda p: DemandCurve(time, c, r, np.append(prices, p), stored, belief1, belief2, belief3, bCap, aversion, volatility, consumption, n, spike) - Generation(consumption, c, time, spike)
    price = sc.bisect(f, -10**10, 10**10)
    deltaStore = DemandCurve(time, c, r, np.append(prices, price), stored, belief1, belief2, belief3, bCap, aversion, volatility, consumption, n, spike) - consumption

    return price, deltaStore


#percentage of opinion 1
def ForecastPercentage(beta, c, r, prices, aversion, cost, n, time, consumption, belief1, belief2, belief3, eta, stored, bCap, spike):
    pi1 = - abs( GetForecast(c, r, prices, 1, aversion, n, volatility, time, consumption, stored[-1], bCap, spike) - prices[-1]) ** (0.2)- 0
    pi2 = - abs( GetForecast(c, r, prices, 2, aversion, n, volatility, time, consumption, stored[-1], bCap, spike) - prices[-1]) ** (0.2)
    pi3 = - abs( GetForecast(c, r, prices, 3, aversion, n, volatility, time, consumption, stored[-1], bCap, spike) - prices[-1]) ** (0.2) - 0
    b1 = eta * (np.exp(beta * pi1) / (np.exp(beta * pi1) + np.exp(beta * pi2) + np.exp(beta * pi3))) + (1 - eta) * belief1[-1]
    b2 = eta * (np.exp(beta * pi2) / (np.exp(beta * pi1) + np.exp(beta * pi2) + np.exp(beta * pi3))) + (1 - eta) * belief2[-1]
    b3 = 1 - b1 - b2
    return b1, b2, b3

#characteristics for aggregate
if (__name__ == '__main__'):
    r = 0.05
    prices = np.array([0, 0])
    prices2 = prices
    belief1 = np.array([1/3])
    belief2 = np.array([1/3])
    belief3 = np.array([1/3])
    bCap = 50000
    aversion = 2
    volatility = 3
    consumption = 20000
    time = 100
    n = 100

    beta = 3
    eta = 0.1
    # cost = 0
    beliefAvg1 = []
    beliefAvg2 = []
    beliefAvg3 = []
    storage = []
    variances = []
    derivatives = []
    c = 200 * np.sin(np.linspace(time/2, time/12, time))
    distrubance = 0.01
    spike = 20000
    np.random.seed(seed=1234)

    #loop over time
    for cost in np.arange(0, 1, 1):
        store = np.array([bCap / 2])
        for i in range(time):
            newPrice, delta = MarketClearing(i, c[i], r, prices, store[-1], belief1[-1], belief2[-1], belief3[-1], bCap, aversion, volatility, consumption, n, spike)
            prices2[-1] = prices[-1] + distrubance
            newPrice2, delta2 = MarketClearing(i, c[i], r, prices2, store[-1], belief1[-1], belief2[-1], belief3[-1], bCap, aversion, volatility, consumption, n, spike)

            prices = np.append(prices, newPrice)
            prices2 = np.append(prices2, newPrice2)
            store = np.append(store, store[-1] + delta)
            newBelief1, newBelief2, newBelief3 = ForecastPercentage(beta, c[i], r, prices, aversion, cost, n, time, consumption, belief1, belief2, belief3, eta, store, bCap, spike)
            belief1 = np.append(belief1, newBelief1)
            belief2 = np.append(belief2, newBelief2)
            belief3 = np.append(belief3, newBelief3)
            # variances = np.append(variances, np.var(prices))

        beliefAvg1 = np.append(beliefAvg1, np.mean(belief1[-time:]))
        beliefAvg2 = np.append(beliefAvg2, np.mean(belief2[-time:]))
        beliefAvg3 = np.append(beliefAvg3, np.mean(belief3[-time:]))
        variances = np.append(variances, np.var(prices[-time:]))
        storage = np.append(storage, store[-time:])
        derivative = np.log(abs((prices2[-time:] - prices[-time:])/distrubance))
        # derivative = derivative[derivative >= -1E308]
        derivatives = np.append(derivatives, np.mean(derivative))
    # plt.plot(derivatives)
    # # plt.yscale('log')
    # plt.xlabel('cost')
    # plt.ylabel('LPE')
    # plt.tight_layout()
    # plt.show()
