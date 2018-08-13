import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc
import math

#market participant
def participant(optionality, r, beta, forecastPrice, prices, stored, opinion, bCap, aversions, volatilities, consumption):
    demand = consumption + (optionality + forecastPrice - (1 + r) * prices[-1] ) / ( 2 * aversions * volatilities )

    if (demand + stored - consumption > bCap):
        demand = bCap - stored - consumption
    elif(demand + stored - consumption < 0):
        demand = consumption - stored

    deltaStore = consumption - demand

    # if (deltaStore > rate * consumption):
    #     demand = -consumption
    #     deltaStore = rate * consumption
    # elif(deltaStore < -rate * consumption):
    #     demand = (rate + 1) * consumption
    #     deltaStore = -rate * consumption

    ruleOneProb = forecastType(optionality, r, beta, prices)
    if ( ruleOneProb > np.random.uniform()):
        opinion = 1
    else:
        opinion = 2

    return demand, stored - deltaStore, opinion

#calculate the demand curve
def demandCurve(volatility, optionality, beta, r, n, storage, generationMat, beliefs, bCap, aversion, pric, p, consumptionMat):
    f = 0
    for i in range(n):
        f = f + participant(optionality, r, beta, forecast(optionality, r, pric, beliefs[i]), np.append(pric, p), storage[i], beliefs[i], bCap[i], aversion[i], volatility[i], consumptionMat[i])[0]
    if f > np.sum(bCap): f = np.sum(bCap)

    f = f - generationMat[-1]
    return f

#find the market clearing price using newtons method
def marketClearing(volatility, beta, n, r, optionality, storage, generationMat, beliefs, bCap, aversion, pric, consumptionMat):
    f = lambda p: demandCurve(volatility, optionality, beta, r, n, storage, generationMat, beliefs, bCap, aversion, pric, p, consumptionMat)
    p = sc.bisect(f, -10**10, 10**10)
    return p

#forecasts for the market price, momentum
def forecast(optionality, r, prices, setting):
    if setting == 1: #trend extrapolation
        return optionality / r +  1.03 * (prices[-2] - optionality / r)
    elif setting == 2: #fundamental price
        return optionality / r

#calculate the probability of choosing rule one
def forecastType(optionality, r, beta, price):
    pi1 = -( forecast(optionality, r, price, 1) - price[-2] ) ** 2
    pi2 = -( forecast(optionality, r, price, 2) - price[-2] ) ** 2
    return 1 / ( 1 + np.exp(beta * ( pi2 - pi1 )) )

def GenerateCharacteristics(n, bCap, aversions, volatilty, consumptions, capacities, prob1):
    bCapMat = np.zeros([n, 1])
    aversionMat = np.zeros([n, 1])
    volatilityMat = np.zeros([n, 1])
    consumptionsMat = np.zeros([n, 1])
    beliefs = np.zeros([n, 1])
    storages = np.zeros([n, 1])

    for i in range(n):
            bCapMat[i] = capacities[np.random.randint(0, len(bCap))]
            aversionMat[i] = aversions[np.random.randint(0, len(aversions))]
            volatilityMat[i] = volatilty[np.random.randint(0, len(volatilty))]
            consumptionsMat[i] = consumptions[np.random.randint(0, len(consumptions))]
            storages[i] = capacities[np.random.randint(0, len(capacities))] / 2 #change initial storage here

            if (np.random.uniform(0, 1) < prob1):
                beliefs[i] = 1
            else:
                beliefs[i] = 2


    return bCapMat, aversionMat, volatilityMat, consumptionsMat, storages, beliefs

#randomly generate generation
#maybe a different function? maybe constraints?
def generation(generationMat, mean):
    if(np.random.uniform(0, 1) < 0.005):
        return mean + np.random.normal(0, 10) # + 0.2  * np.sin(len(generationMat) / 365)
    else:
        return mean #+ 0.2 * np.sin(len(generationMat) / 365)

# -------MAIN-----------
#generate initial prices
n = 10
time = 10000
prob1 = 0.3
capacities = np.array([600, 300, 300])
aversion = np.array([50, 30, 60, 20, 18])
consumption = np.array([30, 50, 30])
volatility = np.array([1.2])
beta = 3
optionality = 2
r = 0.05
percentage = []
totalStorage = []
prices = np.ones([10, 1]) * optionality / r

#per participant
capacities, aversion, volatility, consumptionMat, storage, beliefs = GenerateCharacteristics(n, capacities, aversion, volatility, consumption, capacities, prob1)
generationArray = np.array([np.sum(consumptionMat)])

#5. then loop over things in the correct order and see what happens
for i in range(time):
    consumptionTemp = []
    deltaStoreTemp = []
    settingTemp = []

    for j in range(n):
        consumption, deltaStore, setting = participant(optionality, r, beta, forecast(optionality, r, prices, beliefs[j, -1]), prices, storage[j, -1], beliefs[j, -1], capacities[j], aversion[j], volatility[j], consumptionMat[j])
        consumptionTemp = np.append(consumptionTemp, consumption)
        deltaStoreTemp = np.append(deltaStoreTemp, deltaStore)
        settingTemp = np.append(settingTemp, setting)

        #concatenate temp arrays to full one
    beliefs = np.c_[beliefs, settingTemp]
    storage = np.c_[storage, deltaStoreTemp]

    newPrice = marketClearing(volatility, beta, n, r, optionality, storage[:, i], generationArray, beliefs[:, i], capacities, aversion, prices, consumptionMat)
    generationArray = np.append(generationArray, generation(generationArray, np.sum(consumptionMat))) #with mean
    prices = np.append(prices, newPrice)

for k in range(time):
    percent = (beliefs[:, k] == 1).sum() / len(beliefs[:, 0])
    percentage.append(percent)
    totalStorage.append(np.sum(storage[:,k]))

plt.figure()
plt.subplot(211)
plt.plot(prices)

plt.subplot(212)
plt.plot(percentage)

# plt.subplot(212)
# plt.plot(totalStorage)
plt.show()

#TODO: phase portraits
#TODO: volatility over a moving window
#TODO: bifurcation and stability
#TODO: add noise term for the amateur investors
#TODO: find a better prediction function for seasonality
#TODO: mean time before failure
