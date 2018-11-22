import numpy as np
import scipy.optimize as sc
import matplotlib.pyplot as plt

#demand curve on an aggergate level
def DemandCurve(time, c, r, prices, stored, belief, bCap, aversion, volatility, consumption, n, spike, gamma):
    #get the forecasts first
    push = 5
    # gamma = 0.0001
    forecastPrice1 = GetForecast(c, r, prices, 1,  aversion, n, volatility, time, consumption, stored, bCap, spike, gamma)
    forecastPrice2 = GetForecast(c, r, prices, 2,  aversion, n, volatility, time, consumption, stored, bCap, spike, gamma)

    #find the individual demands for different forecasts
    forecast = belief * forecastPrice1 + (1 - belief) * forecastPrice2
    # - c + 20000
    aggregateDemand = consumption + c + (gamma * (bCap / 2 - stored ) + forecast * (gamma * push + 1) - (1 + r) * prices[-1]) / (2 * aversion * volatility + gamma) #((bCap / 2 - stored - c + push * forecast) + 1000 *(forecast - (1 + r) * prices[-1])) / (2 * aversion * volatility + 1)
    # print(aggregateDemand)
    return aggregateDemand

#generation function
def GetForecast(c, r, prices, belief, aversion, n, volatility, time, consumption, stored, bCap, spike, gamma):
    push = 5
    # gamma = 0.0001
    # + 20000 - c
    fundamentalPrice = ((2 * aversion * volatility + gamma) * (c - Generation(consumption, c, time, spike) + consumption) + gamma * (bCap / 2 - stored)) / (n * r + gamma * push)

    if (belief == 1):
        return fundamentalPrice
    else:
        return fundamentalPrice - 5 * (prices[-2] - fundamentalPrice)

#generation function
def Generation(consumption, c, time, spike):
    if (time == 70): return consumption + c + spike
    if (time == 80): return consumption + c - spike
    return consumption + c #* np.random.normal(0,10)
# + 2 * np.random.uniform(-1,1) +
#market clearing
def MarketClearing(time, c, r, prices, stored, belief, bCap, aversion, volatility, consumption, n, spike, gamma):
    f =  lambda p: DemandCurve(time, c, r, np.append(prices, p), stored, belief, bCap, aversion, volatility, consumption, n, spike, gamma) - Generation(consumption, c, time, spike)
    price = sc.bisect(f, -10**10, 10**10)
    deltaStore = DemandCurve(time, c, r, np.append(prices, price), stored, belief, bCap, aversion, volatility, consumption, n, spike, gamma) - consumption

    return price, deltaStore

#percentage of opinion 1
def ForecastPercentage(beta, c, r, prices, aversion, volatility, cost, n, time, consumption, belief, eta, stored, bCap, spike, gamma):
    pi1 = - ( GetForecast(c, r, prices, 1, aversion, n, volatility, time, consumption, stored, bCap, spike, gamma) - prices[-1]) ** 2 - cost
    pi2 = - ( GetForecast(c, r, prices, 2, aversion, n, volatility, time, consumption, stored, bCap, spike, gamma) - prices[-1]) ** 2

    return eta * 1 / (1 + np.exp(beta * (pi2 - pi1))) + (1 - eta) * belief[-1]

def main(r, beta):
    # r = 0.05
    prices = np.array([0, 0])
    prices2 = prices
    belief = np.array([0.5])
    bCap = 50000
    aversion = 2
    volatility = 3
    consumption = 20000
    time = 5000
    n = 100
    gamma = 0.0001
    # movement = 1.03
    # beta = 0.03
    eta = 0.1
    cost = 1

    beliefAvg = []
    storage = []
    variances = []
    derivatives = []
    c = 200 * np.sin(np.linspace(1, time/12, time))
    distrubance = 0.01
    movement = 1.03
    spike = 0
    start = 0
    store = np.array([bCap / 2])
    #loop over time
    counter = 1
    for i in range(start, start + time):
        # np.random.seed(seed=1234)
        newPrice, delta = MarketClearing(i, c[i], r, prices, store[-1], belief[-1], bCap, aversion, volatility, consumption, n, spike, gamma)
        # prices2[-1] = prices[-1] + distrubance
        # newPrice2, delta2 = MarketClearing(i, c[i], r, prices2, store[-1], belief[-1], bCap, aversion, volatility, consumption, n, movement, spike)
        if(i == 3000):
            cycle = newPrice
            derivative = (newPrice - prices[-1]) / (abs(newPrice - prices[-1]))
            print("the derivative is: ", derivative, " and the calculation is ", (newPrice - prices[-1]) / abs(newPrice - prices[-1]))
        if(i > 3000 and cycle != 0):
            if( abs((cycle - newPrice) / cycle) < 0.01 and (newPrice - prices[-1]) / abs(newPrice - prices[-1]) == derivative):
                break
            counter = counter + 1
        elif(i > 3000):
            if( abs(cycle - newPrice) < 0.0001 and (newPrice - prices[-1]) / abs(newPrice - prices[-1]) == derivative):
                break
            counter = counter + 1

        prices = np.append(prices, newPrice)
        # prices2 = np.append(prices2, newPrice2)
        store = np.append(store, store[-1] + delta)
        belief = np.append(belief, ForecastPercentage(beta, c[i], r, prices, aversion, volatility, cost, n, time, consumption, belief, eta, store[i], bCap, spike, gamma))

        beliefAvg = np.append(beliefAvg, np.mean(belief[-time:]))
        variances = np.append(variances, np.var(prices[-time:]))
        storage = np.append(storage, store[-time:])
        # derivative = np.log(abs((prices2[-1] - prices[-1])/distrubance))

        # derivative = derivative[derivative >= -1E308]
        # derivatives = np.append(derivatives, np.mean(derivative))

    return counter, prices, belief


def Checker(array):
    x = np.diff(array) / array[:-1]
    for i in x:
        if abs(i) > 0.1:
            return True

    return False

#characteristics for aggregate
if (__name__ == '__main__'):
    # beta = 0.03
    # r = 0.01
    values = []
    # counter, prices, belief = main(0.1, 0.05)

    for r in np.arange(0.0001, 0.3, 0.01):
        # for beta in np.arange(0.03, 2.0001, 0.1):
            # for eta in np.arange(0.1, 0.8, 0.1):
        temp = []
        for cost in np.arange(0, 30, 1):
            try:
                counter, prices, belief = main(r, cost)
                temp = np.append(temp, counter)
                print("for r:", r, " and cost: ", cost, " we see derivative: ", counter)
            except:
                print("for r:", r, " and beta:", cost, " we see divergence")
                temp = np.append(temp, np.inf)
                continue
                if(Checker(temp)):
                    jump = np.arange(0, 30, 1)[np.where(np.diff(temp) == np.max(np.diff(temp)))[0][0]]
            values.append(np.asarray([counter, r, cost]))
                # else:
                    # values.append(np.asarray([np.nan, r, beta, eta]))
    plt.imshow(values, interpolation='none', extent=[0.0001, 2.0001, 1.01, 0.01], aspect='auto')
    plt.xlabel('beta')
    plt.ylabel('interest rate')
    plt.colorbar()
    plt.show()
