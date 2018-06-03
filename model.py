import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc
#first pariticipant
def participant1(forecastPrice, prices, stored, opinion):
    bCap = 1000
    aversion = 1.2
    volatility = 0.1
    consumption = 20

    demand = ( forecastPrice - prices[-1] ) / ( aversion * volatility )


    if (demand + stored[-1] - consumption > bCap):
        demand = bCap + stored[-1] - consumption
    elif(demand + stored[-1] - consumption < 0):
        demand = consumption - stored[-1]

    deltaStore = consumption - demand

    changeProb = forecastType(prices, opinion)
    if ( changeProb > 0.3 and opinion == 1):
        opinion = 2
    elif( changeProb > 0.3 and opinion == 2):
        opinion = 1

    return demand, stored[-1] - deltaStore, opinion

#second participant
def participant2(forecastPrice, prices, stored, opinion):
    bCap = 1000
    aversion = 1.2
    volatility = 0.1
    consumption = 15

    demand = ( forecastPrice - prices[-1] ) / ( aversion * volatility )

    if (demand + stored[-1] - consumption > bCap):
        demand = bCap + stored[-1] - consumption
    elif(demand + stored[-1] - consumption < 0):
        demand = consumption - stored[-1]

    deltaStore = consumption - demand

    changeProb = forecastType(prices, opinion)
    if ( changeProb > 0.8 and opinion == 1):
        opinion = 2
    elif( changeProb > 0.8 and opinion == 2):
        opinion = 1

    return demand, stored[-1] - deltaStore, opinion

#randomly generate generation
#maybe a different function? maybe constraints?
def generation(generationMat, mean):
    return  generationMat[-1] + 0.8 * (mean - generationMat[-1]) + np.random.normal(0, 10)

#find the market clearing price using newtons method
#maybe something to restrict if no solution?
def marketClearing(forecastPrice1, forecastPrice2, stored1, stored2, generationMat, beliefs1, beliefs2, pric):
    f = lambda p :  participant1(forecastPrice1, np.append(pric, p), stored1, beliefs1[-1])[0] + participant2(forecastPrice2, np.append(pric, p), stored2, beliefs2[-1])[0] - generationMat[-1]
    p1 = sc.bisect(f, -10**10, 10**10)

    return p1

#forecasts for the market price, momentum
def forecast1(prices):
    return prices[-2] + (prices[-2] - prices[-3])

#forecast for market price, take average of last four time periods
def forecast2(price):
    return np.mean(price[-5:-2])

#calculate the probability of changing
def forecastType(price, setting):
    if ( setting == 1 ):
        prob = np.exp(abs(forecast1(price) - price[-1])) / ( np.exp(abs(forecast1(price) - price[-1])) + abs(forecast2(price) - price[-1]) )
    elif ( setting == 2 ):
        prob = np.exp(abs(forecast2(price) - price[-1])) / ( np.exp(abs(forecast1(price) - price[-1])) + abs(forecast2(price) - price[-1]) )

    return prob

#-------MAIN-----------
#1. generate initial prices
prices = np.array([25, 24, 25, 24, 22])

#2. generate initial generation
generationArray = np.array([100])

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
    storage1 = np.append(storage1, deltaStore1)
    storage2 = np.append(storage2, deltaStore2)

    newPrice = marketClearing(forecast1(prices), forecast2(prices), storage1, storage2, generationArray, belief1, belief2, prices)
    generationArray = np.append(generationArray, generation(generationArray, 40)) #with mean
    prices = np.append(prices, newPrice)

    #ask:
    # 2. changing from one opinion to another: assign randomly, current strategy is fine. generate a random number to see if they change. maybe try a different strategy with storage capacities
    #TODO: implement multiple agents
    #TODO: read over the stuff
