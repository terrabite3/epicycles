import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt











delta = 0.01
xrange = np.arange(-2.25, 0.75, delta)
yrange = np.arange(-1.5, 1.5, delta)
X, Y = np.meshgrid(xrange,yrange)



class Complex:
    def __init__(self, real, img=0):
        self.real = real
        self.img = img

    def __mul__(self, other):
        return Complex(self.real * other.real - self.img * other.img, self.real * other.img + self.img * other.real)

    def __add__(self, other):
        return Complex(self.real + other.real, self.img + other.img)

    def magnitude(self):
        return sqrt(self.real * self.real + self.img * self.img)


def f(n, z):
    if n == 0:
        return Complex(0, 0)
    else:
        return f(n-1, z) * f(n-1, z) + z

Z = Complex(X, Y)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')

# for n in range(8, 9):
for n in [1, 2, 3]:
    print(n)
    F_n = f(n, Z).magnitude()
    ax.contour((F_n - 1), [1])


plt.show()