from scipy.integrate import *
from sympy import *
from numpy import *


# public
def countingH(point):
    k = array(point)
    h = getRating(k)
    return h


# public
def countingHs(points):
    Xs = points[:len(points) / 2]
    Ys = points[len(points) / 2:len(points)]
    hs = []
    for i in xrange(len(Xs)):
        hs = hs + [countingH([Xs[i], Ys[i]])]
    return hs


# private
def getRating(params):
    t = arange(0, 20, 0.01)
    state0 = [3.14, 0, 0, 0]
    state = odeint(pendulumOnCarriage, state0, t, args=(params,))
    return rating(state)


# private
def pendulumOnCarriage(state, t, k):
    a = state[0]
    da = state[1]
    x = state[2]
    dx = state[3]

    M = 1
    m = 0.3
    l = 1
    g = 9.8
    f = 0

    # controller
    x = array([a, da, x, dx])
    f = dot(x, k)

    D = l * M + l * m * (sin(a)) ** 2
    dda = ((M + m) * g * sin(a) - m * l * (da ** 2) * sin(a) * cos(a) - f * cos(a)) / D
    ddx = (m * (l ** 2) * (da ** 2) * sin(a) + l * f - m * g * l * sin(a) * cos(a)) / D
    return [da, dda, dx, ddx]


# private
def rating(state):
    sa = 0
    sDa = 0
    sx = 0
    sDx = 0
    for i in xrange(len(state)):
        a = abs(state[i][0])
        da = abs(state[i][1])
        x = abs(state[i][2])
        dx = abs(state[i][3])

        sa = sa + a
        sDa = sDa + da
        sx = sx + x
        sDx = sDx + dx

    sa = sa / len(state)
    sx = sx / len(state)

    l = len(state)
    sa = sa / l
    sDa = sDa / l
    sx = sx / l
    sDx = sDx / l

    p = (sa * 100 + sx + sDa + sDx)

    return 1. / (p + 1.)
