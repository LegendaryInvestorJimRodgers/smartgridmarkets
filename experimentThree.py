import numpy as np
import scipy.optimize as sc
import matplotlib.pyplot as plt

#demand curve on an aggergate level
def DemandCurve(time, c, r, prices, stored, belief1, belief2, belief3, belief4, belief5, bCap, aversion, volatility, consumption, n):
    #get the forecasts first
    forecastPrice1 = GetForecast(c, r, prices, 1,  aversion, n, volatility, time, consumption)
    forecastPrice2 = GetForecast(c, r, prices, 2,  aversion, n, volatility, time, consumption)
    forecastPrice3 = GetForecast(c, r, prices, 3,  aversion, n, volatility, time, consumption)
    forecastPrice4 = GetForecast(c, r, prices, 4,  aversion, n, volatility, time, consumption)
    forecastPrice5 = GetForecast(c, r, prices, 5,  aversion, n, volatility, time, consumption)

    #find the individual demands for different forecasts
    forecast = belief1 * forecastPrice1 + belief2 * forecastPrice2 + belief3 * forecastPrice3 + belief4 * forecastPrice4 + belief5 * forecastPrice5

    aggregateDemand = consumption + c +  ( forecast - (1 + r) * prices[-1]) / (2 * aversion * volatility)

    #check the constraints for charging
    # if(aggregateDemand + stored - consumption > bCap):
    #     aggergateDemand = consumption - c + bCap - stored
    # elif(aggregateDemand - stored - consumption < 0):
    #     aggregateDemand = consumption - c - stored

    return aggregateDemand

#generation function
def GetForecast(c, r, prices, setting, aversion, n, volatility, time, consumption):
    fundamentalPrice = (2 * aversion * volatility * (c - Generation(consumption, c, time) + consumption)) / (n * r)

    if (setting == 1):
        return fundamentalPrice
    elif(setting == 2):
        return fundamentalPrice + 1.03 * (prices[-2] - fundamentalPrice)
    elif(setting == 3):
        return fundamentalPrice - 1.03 * (prices[-2] - fundamentalPrice)
    elif(setting == 4):
        return fundamentalPrice + 20
    elif(setting == 5):
        return fundamentalPrice - 20
#generation function
def Generation(consumption, c, time):
    if (time == 70): return consumption + c + 20000
    if (time == 80): return consumption + c - 20000
    else: return consumption + c#* np.random.normal(0, 2)

#market clearing
def MarketClearing(time, c, r, prices, stored, belief1, belief2, belief3, belief4, belief5, bCap, aversion, volatility, consumption, n):
    f =  lambda p: DemandCurve(time, c, r, np.append(prices, p), stored, belief1, belief2, belief3, belief4, belief5, bCap, aversion, volatility, consumption, n) - Generation(consumption, c, time)
    price = sc.bisect(f, -10**10, 10**10)
    deltaStore = DemandCurve(time, c, r, np.append(prices, price), stored, belief1, belief2, belief3, belief4, belief5, bCap, aversion, volatility, consumption, n) - consumption

    return price, deltaStore

#percentage of opinion 1
def ForecastPercentage(beta, c, r, prices, aversion, cost, n, time, consumption, belief1, belief2, belief3, belief4, belief5, eta):
    pi1 = - abs( GetForecast(c, r, prices, 1, aversion, n, volatility, time, consumption) - prices[-1]) ** (0.2)- cost
    pi2 = - abs( GetForecast(c, r, prices, 2, aversion, n, volatility, time, consumption) - prices[-1]) ** (0.2)
    pi3 = - abs( GetForecast(c, r, prices, 3, aversion, n, volatility, time, consumption) - prices[-1]) ** (0.2)
    pi4 = - abs( GetForecast(c, r, prices, 4, aversion, n, volatility, time, consumption) - prices[-1]) ** (0.2)
    pi5 = - abs( GetForecast(c, r, prices, 5, aversion, n, volatility, time, consumption) - prices[-1]) ** (0.2)
    b1 = eta * (np.exp(beta * pi1) / (np.exp(beta * pi1) + np.exp(beta * pi2) + np.exp(beta * pi3) + np.exp(beta * pi4) + np.exp(beta * pi5))) + (1 - eta) * belief1[-1]
    b2 = eta * (np.exp(beta * pi2) / (np.exp(beta * pi1) + np.exp(beta * pi2) + np.exp(beta * pi3) + np.exp(beta * pi4) + np.exp(beta * pi5))) + (1 - eta) * belief2[-1]
    b3 = eta * (np.exp(beta * pi3) / (np.exp(beta * pi1) + np.exp(beta * pi2) + np.exp(beta * pi3) + np.exp(beta * pi4) + np.exp(beta * pi5))) + (1 - eta) * belief3[-1]
    b4 = eta * (np.exp(beta * pi4) / (np.exp(beta * pi1) + np.exp(beta * pi2) + np.exp(beta * pi3) + np.exp(beta * pi4) + np.exp(beta * pi5))) + (1 - eta) * belief4[-1]
    b5 = 1 - b1 - b2 - b3 - b4
    return b1, b2, b3, b4, b5

#characteristics for aggregate
if (__name__ == '__main__'):
    r = 0.05
    prices = np.array([1, -1])
    prices2 = prices
    belief1 = np.array([1/5])
    belief2 = np.array([1/5])
    belief3 = np.array([1/5])
    belief4 = np.array([1/5])
    belief5 = np.array([1/5])
    belief1Avg = []
    belief2Avg = []
    belief3Avg = []
    belief4Avg = []
    belief5Avg = []
    bCap = 50000
    aversion = 2
    volatility = 3
    consumption = 20000
    time = 1000
    n = 100

    # beta = 1
    eta = 0.1
    cost = 1
    storage = []
    variances = []
    derivatives = []
    recoveryTime = []
    spikeSize = []
    c = 200 * np.sin(np.linspace(1, time/12, time))
    distrubance = 0.01
    counter = 0

    #loop over time
    for beta in np.arange(0.03, 20, 0.4):
        store = np.array([bCap / 2])
        for i in range(time):
            newPrice, delta = MarketClearing(i, c[i], r, prices, store[-1], belief1[-1], belief2[-1], belief3[-1], belief4[-1], belief5[-1], bCap, aversion, volatility, consumption, n)
            prices2[-1] = prices[-1] + distrubance
            newPrice2, delta2 = MarketClearing(i, c[i], r, prices2, store[-1], belief1[-1], belief2[-1], belief3[-1], belief4[-1], belief5[-1], bCap, aversion, volatility, consumption, n)

            prices = np.append(prices, newPrice)
            prices2 = np.append(prices2, newPrice2)
            store = np.append(store, store[-1] + delta)
            newBelief1, newBelief2, newBelief3, newBelief4, newBelief5 = ForecastPercentage(beta, c[i], r, prices, aversion, cost, n, time, consumption, belief1, belief2, belief3, belief4, belief5, eta)
            belief1 = np.append(belief1, newBelief1)
            belief2 = np.append(belief2, newBelief2)
            belief3 = np.append(belief3, newBelief3)
            belief4 = np.append(belief4, newBelief4)
            belief5 = np.append(belief5, newBelief5)

        variances = np.append(variances, np.var(prices[-time:]))
        storage = np.append(storage, store[-time:])
        derivative = np.log(abs((prices2[-time:] - prices[-time:])/distrubance))
        belief1Avg = np.append(belief1Avg, np.mean(belief1[-time:]))
        belief2Avg = np.append(belief2Avg, np.mean(belief2[-time:]))
        belief3Avg = np.append(belief3Avg, np.mean(belief3[-time:]))
        belief4Avg = np.append(belief4Avg, np.mean(belief4[-time:]))
        belief5Avg = np.append(belief5Avg, np.mean(belief5[-time:]))
        recoveryTime = np.append(recoveryTime, np.where(abs(prices[(72 + counter * cost):]) < 0.5)[0][0])
        spikeSize = np.append(spikeSize, np.max(abs(prices[(cost * time):])))
        # derivative = derivative[derivative >= -1E308]
        derivatives = np.append(derivatives, np.mean(derivative))
        counter = counter + 1
    plt.plot(np.arange(0.1, 20, 0.4), derivatives)
    # plt.yscale('log')
    plt.xlabel('beta')
    plt.ylabel('LPE')
    plt.tight_layout()
    plt.show()
