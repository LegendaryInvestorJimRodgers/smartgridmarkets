import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc
import math
from PIL import Image


#market participant
def participant(optionality, r, beta, forecastPrice, prices, stored, opinion, bCap, aversions, volatilities, consumption, generation, n, time):
    gamma = 0.0001
    push = 5
    demand = consumption + (gamma * (bCap / 2 - stored) + forecastPrice * (gamma * push + 1) - (1 + r) * prices[-1]) / (2 * aversions * volatilities + gamma)

    if (bCap == 0): demand = consumption
    # if(demand + stored - consumption > bCap):
    #     demand = consumption  + bCap - stored
    # elif(demand - stored - consumption < 0):
    #     demand = consumption  - stored

    # if (demand + stored - consumption > bCap):
    #     demand = bCap - stored - consumption
    # elif(demand + stored - consumption < 0):
    #     demand = consumption - stored

    deltaStore = consumption - demand

    ruleOneProb = forecastType(optionality, r, beta, prices, consumption, generation, n, time)

    if ( ruleOneProb > 0.5):
        opinion = 1
    else:
        opinion = 2

    return demand, stored - deltaStore, opinion

#calculate the demand curve
def demandCurve(volatility, optionality, beta, r, n, storage, generationMat, beliefs, bCap, aversion, pric, p, consumptionMat, time):
    f = 0

    for i in range(n):
        f = f + participant(optionality, r, beta, forecast(optionality, r, pric, beliefs[i], consumptionMat[i], generationMat[-1], n, time), np.append(pric, p), storage[i], beliefs[i], bCap[i], aversion[i], volatility[i], consumptionMat[i], generationMat[-1], n, time)[0]
    # if f > np.sum(bCap): f = np.sum(bCap)

    f = f - generationMat[-1]
    return f

#find the market clearing price using newtons method
def marketClearing(volatility, beta, n, r, optionality, storage, generationMat, beliefs, bCap, aversion, pric, consumptionMat, time):
    f = lambda p: demandCurve(volatility, optionality, beta, r, n, storage, generationMat, beliefs, bCap, aversion, pric, p, consumptionMat, time)
    p = sc.bisect(f, -10**10, 10**10)
    return p

#forecasts for the market price, momentum
def forecast(optionality, r, prices, setting, consumption, generationMat, n, i):
    gamma = 0.0001
    push = 5
    fundamentalPrice = ((2 * aversion[0] * volatility[0] + gamma) * (consumption - generation(generationMat, consumption, i) + 2 * np.sin(i / 12)) + gamma * (bCapTotal / 2 - storageTotal)) / (n * r - n * gamma * push)

    if setting == 1: #trend extrapolation
        return fundamentalPrice
    elif setting == 2: #fundamental price
        return fundamentalPrice +  1.01 * (prices[-2] - fundamentalPrice)

#calculate the probability of choosing rule one
def forecastType(optionality, r, beta, price, consumption, generationMat, n, time):
    pi1 = -( forecast(optionality, r, price, 1, consumption, generationMat, n, time) - price[-2]) ** 2 - 0.2
    pi2 = -( forecast(optionality, r, price, 2, consumption, generationMat, n, time) - price[-2]) ** 2
    return 1 / ( 1 + np.exp(beta * ( pi2 - pi1 )) )

def GenerateCharacteristics(n, bCap, aversions, volatilty, consumptions, capacities, prob1, totCap, percent):
    bCapMat = np.zeros([n, 1])
    aversionMat = np.zeros([n, 1])
    volatilityMat = np.zeros([n, 1])
    consumptionsMat = np.zeros([n, 1])
    beliefs = np.zeros([n, 1])
    storages = np.zeros([n, 1])

    for i in range(n):
            if (i < n * percent): bCapMat[i] = totCap / (n * percent)
            if (i >= n * percent): bCapMat[i] = 0
            aversionMat[i] = aversions[np.random.randint(0, len(aversions))]
            volatilityMat[i] = volatilty[np.random.randint(0, len(volatilty))]
            consumptionsMat[i] = consumptions[np.random.randint(0, len(consumptions))]
            storages[i] =  bCapMat[i] / 2 #change initial storage here

            if (0.5 < prob1):
                beliefs[i] = 1
            else:
                beliefs[i] = 2


    return bCapMat, aversionMat, volatilityMat, consumptionsMat, storages, beliefs

#randomly generate generation
#maybe a different function? maybe constraints?
def generation(generationMat, mean, i):
    # if(np.random.uniform(0, 1) < 0.01):
    # c = 2 * np.sin(i / 8)
    c= 0
    if(i == 70):
        return mean + c + 20000# + 200 * np.random.normal(0, 10) # + 0.2  * np.sin(len(generationMat) / 365)
    elif(i == 50):
        return mean + c - 20000
    else:
        return mean + c #+ 0.2 * np.sin(len(generationMat) / 365)

# -------MAIN-----------
#generate initial prices
#per participant
variances = []
values = []
for k in range(1, 101, 10):
    n = 100
    time = 100
    prob1 = 0.5
    capacities = np.array([400, 250])
    aversion = np.array([2])
    consumption = np.array([40])
    volatility = np.array([3])
    beta = 0.03
    optionality = 2
    r = 0.02
    percentage = []
    totalStorage = []
    prices = np.ones([2, 1]) * 0
    np.random.seed(seed = 11234)
    bCapTotal = 50000

    capacities, aversion, volatility, consumptionMat, storage, beliefs = GenerateCharacteristics(n, capacities, aversion, volatility, consumption, capacities, prob1, 50000, k / 100)
    generationArray = np.array([np.sum(consumptionMat)])

    #5. then loop over things in the correct order and see what happens
    for i in range(time):
        consumptionTemp = []
        deltaStoreTemp = []
        settingTemp = []
        storageTotal = np.sum(storage[:, -1])

        for j in range(n):

            consumption, deltaStore, setting = participant(optionality, r, beta, forecast(optionality, r, prices, beliefs[j, -1], consumptionMat[j], generationArray[-1], n, i), prices, storage[j, -1], beliefs[j, -1], capacities[j], aversion[j], volatility[j], consumptionMat[j], generationArray[-1], n, i)
            consumptionTemp = np.append(consumptionTemp, consumption)
            deltaStoreTemp = np.append(deltaStoreTemp, deltaStore)
            settingTemp = np.append(settingTemp, setting)

            #concatenate temp arrays to full one
        beliefs = np.c_[beliefs, settingTemp]
        storage = np.c_[storage, deltaStoreTemp]

        newPrice = marketClearing(volatility, beta, n, r, optionality, storage[:, i], generationArray, beliefs[:, i], capacities, aversion, prices, consumptionMat, i)
        generationArray = np.append(generationArray, generation(generationArray, np.sum(consumptionMat), i)) #with mean
        prices = np.append(prices, newPrice)

    print(np.min(prices))
    print(np.max(prices))
    variances = np.append(variances, np.var(prices))
    for k in range(time):
        percent = (beliefs[:, k] == 1).sum() / len(beliefs[:, 0])
        percentage.append(percent)
        totalStorage.append(np.sum(storage[:,k]))

    values.append(np.unique(np.floor(prices)))

imgx = 601
imgy =  100
increment = 1
count = 0
array = np.asarray(values)
min = np.min(array[0])
max = np.max(array[0])
array = array - min

imgx = imgy / increment - 1
imgy = max - min + 1
image = Image.new("RGB", (int(imgx), int(imgy)))
array = array[1:]

for i in range(int(imgx)):
    for j in array[i]:
        image.putpixel((i, int(j)), (255, 255, 255))
image.show()
