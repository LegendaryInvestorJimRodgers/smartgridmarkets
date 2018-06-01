import numpy as np

#1. declare starting variables
startPrices = (np.random.rand(10,1) - 0.5) * 3
volatility = 2
riskFree = 0.05

#2. start the forecasting model
# there will be two predictions, simeple average, or just the last value
def predictions(prices):
    return prices[-1], np.mean(prices)

#3. calculate the relative performance of each model
def performanceCheck(pred, prices, riskFree):
    return pred - (1 + riskFree / (365 * 24)) * prices[-2]

#4. market clearing
def clearPrice(numSetting0, numSetting1, pred0, pred1):
    return numSetting0 * pred0 + numSetting1 * pred1


#5. create agents
def main(numSetting0, numSetting1, startPrices = startPrices, volatility = volatility, riskFree = riskFree, iterations = 10):
    price = startPrices
    pred = []
    performance1 = []
    performance2 = []

    for i in range(iterations):
        pred0, pred1 = predictions(0, price)

        #calculate clearing
        prices.append(clearPrice(numSetting0, numSetting1, pred0, pred1))

        #calculate performance and adjust behavior
        performance1.append(performanceCheck(pred0, prices, riskFree))
        performance2.append(performanceCheck(pred0, prices, riskFree))

        if(performance1[-1] > performance2[-1]):

#6. create an actual agent that changes his opinion
def agents():
