import numpy as np
import scipy.optimize as sc
import matplotlib.pyplot as plt
from PIL import Image

#demand curve on an aggergate level
def DemandCurve(time, c, r, prices, stored, belief, bCap, aversion, volatility, consumption, n, spike):
    #get the forecasts first
    push = 5
    gamma = 0.003
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
    gamma = 0.003
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
    cost = 2
    variances = []
    c = 2 * np.sin(np.linspace(1, time/12, time))
    values = []
    imgx = 1000
    imgy =  50
    increment = 1
    #loop over time
    for spike in np.arange(1, imgy, increment):
        prices = np.array([0, 0])
        store = np.array([bCap / 2])
        for i in range(time):
            newPrice, delta = MarketClearing(i, c[i], r, prices, store[-1], belief[-1], bCap, aversion, volatility, consumption, n, spike)

            prices = np.append(prices, newPrice)
            store = np.append(store, store[-1] + delta)
            belief = np.append(belief, ForecastPercentage(beta, c[i], r, prices, aversion, cost, n, time, consumption, belief, eta, store[-1], bCap, spike))
            # variances = np.append(variances, np.var(prices))

        values.append(np.unique(np.floor(prices)))

    count = 0
    array = np.asarray(values)
    min = np.min(array[-1])
    max = np.max(array[-1])
    array = array - min

    imgx = imgy / increment - 1
    imgy = max - min + 1
    image = Image.new("RGB", (int(imgx), int(imgy)))

    for i in range(int(imgx)):
        for j in array[i]:
            image.putpixel((i, int(j)), (255, 255, 255))
    image.show()
