import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Initializing number of dots
N = 25
XC = 5
YC = 5
R = 1
R_SQR = R **2

# Creating dot class
class Dot(object):
    def __init__(self, ind):
        self.x = 10 * np.random.random_sample()
        self.y = 10 * np.random.random_sample()
        self.velx = self.generate_new_vel()
        self.vely = self.generate_new_vel()

    def generate_new_vel(self):
        # rand = np.random.random_sample() - 0.5 / 5
        # if rand > 0.5:
        #     return v + rand/ 10.
        # else:
        #     return v - rand/10.
        return  (np.random.random_sample() - 0.5) / 5

    def move(self) :
        if np.random.random_sample() < 0.95:
            self.x += self.velx
            self.y += self.vely
        else:
            self.velx = self.generate_new_vel()
            self.vely = self.generate_new_vel()
            self.x += self.velx
            self.y += self.vely
        # self.check_inside_circle()


# Initializing dots
dots = [Dot(i) for i in range(N)]

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
d, = ax.plot([dot.x for dot in dots],
             [dot.y for dot in dots], 'ro', markersize=3)
# circl# Initializing dots
dots = [Dot(i) for i in range(N)]
e = plt.Circle((XC, YC), R, color='b', fill=False)
# ax.add_artist(circle)


# animation function.  This is called sequentially
def animate(i):
    for dot in dots:
        dot.move()
    d.set_data([dot.x for dot in dots],
               [dot.y for dot in dots])
    return d,

# call the animator.
anim = animation.FuncAnimation(fig, animate, frames=200, interval=20)
# plt.axis('equal')
plt.show()
