import numpy as np
import matplotlib.pyplot as plt

#first pariticipant
def participant1(forecastPrice, prices, stored, opinion):
    bCap = 1000
    aversion = 1.2
    volatility = 0.1
    consumption = 20

    demand = ( forecastPrice - prices[-1] ) / ( aversion * volatility )
    deltaStore = consumption - demand

    if (demand + stored[-1] - consumption > bCap):
        demand = bcap + stored[-1] - consumption
    elif(demand + stored[-1] - consumption < 0):
        demand = consumption - stored[-1]

    deltaStore = consumption - demand

    changeProb = forecastType(prices, opinion)
    if ( changeProb > 0.5 and opinion == 1):
        opinion = 2
    elif( changeProb > 0.5 and opinion == 2):
        opinion = 1

    return demand, stored[-1] - deltaStore , opinion

#second participant
def participant2(forecastPrice, prices, stored, opinion):
    bCap = 1000
    aversion = 1.2
    volatility = 0.1
    consumption = 20

    demand = ( forecastPrice - prices[-1] ) / ( aversion * volatility )
    deltaStore = consumption - demand

    if (demand + stored[-1] - consumption > bCap):
        demand = bcap + stored[-1] - consumption
    elif(demand + stored[-1] - consumption < 0):
        demand = consumption - stored[-1]

    deltaStore = consumption - demand

    changeProb = forecastType(prices, opinion)
    if ( changeProb > 0.5 and opinion == 1):
        opinion = 2
    elif( changeProb > 0.5 and opinion == 2):
        opinion = 1

    return demand, stored[-1] - deltaStore, opinion

#randomly generate generation
#maybe a different function? maybe constraints?
def generation(generationMat):
    return generationMat[-1] + np.random.normal(0, 10)

#find the market clearing price using newtons method
#maybe something to restrict if no solution?
def marketClearing(forecastPrice1, forecastPrice2, stored1, stored2, generationMat, beliefs1, beliefs2, pric):
    tolerance = 0.00001
    dp = 0.01
    p0 = 10
    f = lambda p :  participant1(forecastPrice1, np.append(pric, p), stored1, beliefs1[-1])[0] + participant2(forecastPrice2, np.append(pric, p), stored2, beliefs2[-1])[0] - generationMat[-1]
    f0 = f(p0)
    p1 = p0 - f0 / ( (f0 - f(p0 + dp)) / dp )

    while(abs(p1 - p0) / p0 > tolerance):
        p0 = p1
        f0 = f(p0)
        p1 = p0 - f0 / ( (f0 - f(p0 + dp)) / dp )
        # print(p1)
    print(p1)
    return p1

#forecasts for the market price, take latest price
#WARNING: CHECK THAT -2 STUFF FOR THE PRICES!
def forecast1(prices):
    return prices[-2]

#forecast for market price, take average of last four time periods
def forecast2(price):
    return np.mean(price[-4:-1])

#calculate the probability of changing
def forecastType(price, setting):
    if ( setting == 1 ):
        prob = np.exp(abs(forecast1(price) - price[-1])) / ( np.exp(abs(forecast1(price) - price[-1])) + abs(forecast2(price) - price[-1]) )
    elif ( setting == 2 ):
        prob = np.exp(abs(forecast2(price) - price[-1])) / ( np.exp(abs(forecast1(price) - price[-1])) + abs(forecast2(price) - price[-1]) )

    return prob

#-------MAIN-----------
#1. generate initial prices
prices = np.array([-25, -24, -25, -24])

#2. generate initial generation
generationArray = np.array([70])

#3. generate initial beliefs
belief1 = np.array([1])
belief2 = np.array([2])

#4. generate intiial storage
storage1 = np.array([200])
storage2 = np.array([500])

#5. then loop over things in the correct order and see what happens
for i in range(100):
    if ( belief1[-1] == 1 ):
        consumption1, deltaStore1, setting1 = participant1(forecast1(prices), prices, storage1, 1)
    elif ( belief1[-1] == 2 ):
        consumption1, deltaStore1, setting1 = participant1(forecast2(prices), prices, storage1, 2)

    if ( belief2[-1] == 1 ):
        consumption2, deltaStore2, setting2 = participant2(forecast1(prices), prices, storage2, 1)
    elif( belief2[-1] == 2 ):
        consumption2, deltaStore2, setting2 = participant2(forecast2(prices), prices, storage2, 2)

    belief1 = np.append(belief1, setting1)
    belief2 = np.append(belief2, setting2)
    storage1 = np.append(storage1, storage1[-1] - deltaStore1)
    storage2 = np.append(storage2, storage2[-1] - deltaStore2)

    newPrice = marketClearing(forecast1(prices), forecast2(prices), storage1, storage2, generationArray, belief1, belief2, prices)
    generationArray = np.append(generationArray, generation(generationArray))
    prices = np.append(prices, newPrice)

    #ask:
    # 1. root finding
    # 2. changing from one opinion to another
    # 3. what types of forecasts should i use?
    # 4. starting data
    # 5. 
