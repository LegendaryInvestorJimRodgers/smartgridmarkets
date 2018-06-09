import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc

#market participant
def participant(forecastPrice, prices, stored, opinion, bCap, aversions, volatilities, consumption, probability):
    demand = ( forecastPrice - prices[-1] ) / ( aversions * volatilities )

    if (demand + stored - consumption > bCap):
        demand = bCap - stored - consumption
        # print("the demand is:", demand)
        # print("the capacity is:", bCap)
        # print("the storage is:", stored)
        # print("the consumption is:", consumption)
        # print("the forecast price is:", forecastPrice)
    elif(demand + stored - consumption < 0):
        demand = consumption - stored

    deltaStore = consumption - demand

    changeProb = forecastType(prices, opinion)
    if ( changeProb > probability and opinion == 1):
        opinion = 2
    elif( changeProb > probability and opinion == 2):
        opinion = 1

    return demand, stored - deltaStore, opinion

#calculate the demand curve
def demandCurve(n, storage, generationMat, beliefs, bCap, aversion, pric, p, probability, consumptionMat):
    f = 0
    for i in range(n):
        f = f + participant(forecast(pric, beliefs[i]), np.append(pric, p), storage[i], beliefs[i], bCap[i], aversion[i], volatility[i], consumptionMat[i], probability[i])[0]

    f = f - generationMat[-1]
    return f

#find the market clearing price using newtons method
def marketClearing(n, storage, generationMat, beliefs, bCap, aversion, pric, probability, consumptionMat):
    f = lambda p: demandCurve(n, storage, generationMat, beliefs, bCap, aversion, pric, p, probability, consumptionMat)
    p = sc.bisect(f, -10**10, 10**10)

    return p

#forecasts for the market price, momentum
def forecast(prices, setting):
    if setting == 1: #trend extrapolation
        return prices[-2] + (prices[-2] - prices[-3])
    elif setting == 2: #average of four prices
        return np.mean(prices[-5:-2])


#calculate the probability of changing
def forecastType(price, setting):
    return np.exp(abs(forecast(price, setting) - price[-1])) / ( np.exp(abs(forecast(price, setting) - price[-1])) + abs(forecast(price, setting) - price[-1]) )

#generate beliefs here
def GenerateBeliefs(n, prob1):
    beliefs = np.zeros([n, 1])
    for i in range(n):
        if (np.random.uniform(0, 1) < prob1):
            beliefs[i] = 1
        else:
            beliefs[i] = 2

    return beliefs

#generate storage here
def GenerateStorage(n, capacities):
    storages = np.zeros([n, 1])
    for i in range(n):
            storages[i] = capacities[np.random.randint(0, len(capacities))] / 2

    return storages

def GenerateCharacteristics(n, bCap, aversions, volatilty, consumptions, probabilities):
    bCapMat = np.zeros([n, 1])
    aversionMat = np.zeros([n, 1])
    volatilityMat = np.zeros([n, 1])
    consumptionsMat = np.zeros([n, 1])
    probabilitiesMat = np.zeros([n, 1])

    for i in range(n):
            bCapMat[i] = capacities[np.random.randint(0, len(bCap))]
            aversionMat[i] = aversion[np.random.randint(0, len(aversion))]
            volatilityMat[i] = volatilty[np.random.randint(0, len(volatilty))]
            consumptionsMat[i] = consumptions[np.random.randint(0, len(consumptions))]
            probabilitiesMat[i] = probabilities[np.random.randint(0, len(probabilities))]

    return bCapMat, aversionMat, volatilityMat, consumptionsMat, probabilitiesMat

#randomly generate generation
#maybe a different function? maybe constraints?
def generation(generationMat, mean):
    return  generationMat[-1] + 0.8 * (mean - generationMat[-1]) + np.random.normal(0, 10)

#-------MAIN-----------
#generate initial prices
prices = np.array([25, 24, 25, 24, 22])
n = 50
time = 100
prob1 = 0.3
capacities = np.array([500, 200, 300, 600])
aversion = np.array([5, 3, 6, 2, 18])
consumption = np.array([20, 30, 10])
probabilities = np.array([0.3, 0.5, 0.8, 0.4])
volatility = np.array([1.2])

#generate beliefs
beliefs = GenerateBeliefs(n, prob1)

#4. generate intiial storage
storage = GenerateStorage(n, capacities)

#per participant
capacities, aversion, volatility, consumptionMat, probabilities = GenerateCharacteristics(n, capacities, aversion, volatility, consumption, probabilities)

#mean generation is the average consumption
meanGeneration = np.mean(consumption)
generationArray = np.array([n * meanGeneration])

#5. then loop over things in the correct order and see what happens
for i in range(time):
    consumptionTemp = []
    deltaStoreTemp = []
    settingTemp = []

    for j in range(n):
        consumption, deltaStore, setting = participant(forecast(prices, beliefs[j, -1]), prices, storage[j, -1], beliefs[j, -1], capacities[j], aversion[j], volatility[j], consumptionMat[j], probabilities[j])
        consumptionTemp = np.append(consumptionTemp, consumption)
        deltaStoreTemp = np.append(deltaStoreTemp, deltaStore)
        settingTemp = np.append(settingTemp, setting)

    #concatenate temp arrays to full one
    beliefs = np.c_[beliefs, settingTemp]
    storage = np.c_[storage, deltaStoreTemp]

    newPrice = marketClearing(n, storage[:, i], generationArray, beliefs[:, i], capacities, aversion, prices, probabilities, consumptionMat)
    generationArray = np.append(generationArray, generation(generationArray, n * meanGeneration)) #with mean
    prices = np.append(prices, newPrice)

#TODO: the fundamental price is when the forecast is the same as the realized price. We can for now assume that the fundamental price is fixed (fixed generation too)
#TODO: read the papers on real options
#TODO: implement the fixed withdrawal and injection rates
