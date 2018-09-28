import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

#plotting the pay-off
data = pd.read_csv('BatteryPrices6.csv')
df = data.unstack().reset_index()
df.columns = ['X', 'Y', 'Z']
df['Y'] =  np.tile(range(-500, 3001, 50), 21)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(df['Y'], pd.to_numeric(df['X']), df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
plt.ylabel('Starting Storage')
plt.xlabel('Starting Price')

# plotting the pay-off value
prices = pd.read_csv('BatteryPrices6Der.csv')
prices = pd.DataFrame(prices).unstack(level=0).reset_index()
prices.columns = ['X', 'Y', 'Z']
prices['Y'] =  np.tile(range(-500, 3001, 50), 20)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(prices['Y'], pd.to_numeric(prices['X']), prices['Z'], cmap=plt.cm.viridis, linewidth=0.2)
plt.ylabel('Starting Storage')
plt.xlabel('Starting Price')
plt.show()
