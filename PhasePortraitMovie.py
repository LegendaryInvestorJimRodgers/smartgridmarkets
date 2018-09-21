import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fut = 0
cons = 40.1
gen = 40
a = 10
V = 3
n1 = 0.5
n = 10
r = 0.05
b = 1.12
beta = 0.5
eta = 0.5
cost = 0
time = 1000

startP = -50
endP = 50
tickP = 30
startO = 0
endO = 1
tickO = 30

X, Y = np.mgrid[-10:70:5j, 0:1:10j]
fund = (2 * a * V * (fut - gen + cons)) / (n * r)
pi1 = - (fund - X) ** 2 - cost
pi2 = - (fund + b * (X - fund))
U = ((gen - cons - fut) * 2 * a * V - Y * fund - (1 - Y) * fund + (1 - Y) * b * fund) / ((1 - Y) * b - (1 + r))
V = 1 / (1 + np.exp(beta * pi2 - beta * pi1))

fig, ax = plt.subplots(1,1)
Q = ax.quiver(X, Y, U, V, pivot='mid', color='r', units='inches')

ax.set_xlim(-1, 7)
ax.set_ylim(-1, 7)

def update_quiver(num, Q, X, Y):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """

    U = np.cos(X + num*0.1)
    V = np.sin(Y + num*0.1)

    Q.set_UVC(U,V)

    return Q,

# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y),
                               interval=50, blit=False)
fig.tight_layout()
plt.show()
