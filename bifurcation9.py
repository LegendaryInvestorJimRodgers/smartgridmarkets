import numpy as np
import scipy.optimize as sc
import matplotlib.pyplot as plt
from PIL import Image

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

    belief1 = np.array([1/5])
    belief2 = np.array([1/5])
    belief3 = np.array([1/5])
    belief4 = np.array([1/5])
    belief5 = np.array([1/5])
    bCap = 50000
    aversion = 2
    volatility = 3
    consumption = 20000
    time = 100
    n = 100
    store = np.array([bCap / 2])
    # beta = 0.03
    eta = 0.1
    cost = 2
    variances = []
    c = 200 * np.sin(np.linspace(1, time/12, time))
    values = []
    imgx = 1000
    imgy =  20
    increment = 0.1
    cost = 1
    #loop over time
    for beta in np.arange(0.01, imgy, increment):
        prices = np.array([0, 0])
        store = np.array([bCap / 2])
        for i in range(time):
            newPrice, delta = MarketClearing(i, c[i], r, prices, store[-1], belief1[-1], belief2[-1], belief3[-1], belief4[-1], belief5[-1], bCap, aversion, volatility, consumption, n)

            prices = np.append(prices, newPrice)
            store = np.append(store, store[-1] + delta)
            newBelief1, newBelief2, newBelief3, newBelief4, newBelief5 = ForecastPercentage(beta, c[i], r, prices, aversion, cost, n, time, consumption, belief1, belief2, belief3, belief4, belief5, eta)
            belief1 = np.append(belief1, newBelief1)
            belief2 = np.append(belief2, newBelief2)
            belief3 = np.append(belief3, newBelief3)
            belief4 = np.append(belief4, newBelief4)
            belief5 = np.append(belief5, newBelief5)
            # variances = np.append(variances, np.var(prices))

        values.append(np.unique(np.floor(prices / 1000)))

    count = 0
    array = np.asarray(values)
    min = np.min(array[0])
    max = np.max(array[0])
    array = array + 300

    imgx = imgy / increment - 1
    imgy = max + 400 + 1
    image = Image.new("RGB", (int(imgx), int(imgy)))

    for i in range(int(imgx)):
        for j in array[i]:
            print(j)
            image.putpixel((i, int(j)), (255, 255, 255))
    image.show()
