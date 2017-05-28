import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from math import *


def load_datasets(n=20, pointsInFrame=100):
    x = []
    y = []
    for i in xrange(n):
        data_frame_X = load_data_frame_X(countPoints=pointsInFrame)
        x = x + list(data_frame_X)
        y = y + list(load_data_frame_Y(data_frame_X))
    return [x, y]


def load_dataset_uniform(n=20, step=0.01):
    x = []
    y = []
    for i in xrange(n):
        point = list(random.rand(2))
        data_frame_X = load_dataset_uniform_frame(step=step, centrPoint=point)
        x = x + list(data_frame_X)
        y = y + [point]
    return [x, y]


def load_dataset_uniform_frame(step=0.01, centrPoint=[0.5, 0.5]):
    box = np.arange(0, 1, step)

    y = []
    x = []
    for i in box:
        y = y + list(np.linspace(i, i, len(box)))
        x = x + list(box)

    h = []
    for i in xrange(len(x)):
        h = h + [countingH([x[i], y[i]], centrPoint)]

    return [x + y + h]


def load_data_frame_X(countPoints=3):
    centPoint = list(random.rand(2))
    x = list(random.rand(countPoints))
    y = list(random.rand(countPoints))
    h = []
    for i in xrange(countPoints):
        h = h + [countingH([x[i], y[i]], centPoint)]
    return [x + y + h]


def load_data_frame_Y(data_frame_X):
    maxH = -10
    sX = 0
    sY = 0

    data = data_frame_X[0]
    for i in xrange(len(data) / 3):
        if (data[3 * i + 2] > maxH):
            maxH = data[3 * i + 2]
            sX = data[3 * i]
            sY = data[3 * i + 1]

    return [[sX, sY]]


def color(point, centPoint):
    # c = [0.78, 0.7]
    d = sqrt((point[0] - centPoint[0]) ** 2 + (point[1] - centPoint[1]) ** 2)
    red = (1 - d) ** 2
    blue = d / 5
    if (red < 0):
        red = 0
    if (blue > 1):
        blue = 1
    if (blue < 0):
        blue = 0
    return [red, 0, blue]


def countingH(point, centrPoint):
    # c = [0.78, 0.7]
    d = sqrt((point[0] - centrPoint[0]) ** 2 + (point[1] - centrPoint[1]) ** 2)
    h = (1 - d) ** 2
    return h


def draw_model(centrPoint, predictPoint=[-1, -1], fileNameToSave='draw_model', show=1):
    box = np.arange(0, 1, 0.01)

    y = []
    x = []
    for i in box:
        y = y + list(np.linspace(i, i, len(box)))
        x = x + list(box)

    colors = []
    for i in xrange(len(x)):
        colors.append(color([x[i], y[i]], centrPoint))

    xx = x
    yy = y

    x = np.asarray(xx)
    y = np.asarray(yy)

    # colors = x-y
    # area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii

    # x = x * 100 - 50;
    # y = y * 100 - 50;

    soa = np.array([[0, 0, 0.3, 30], [0, 0, 6, 3]])
    X, Y, U, V = zip(*soa)
    plt.figure()
    ax = plt.gca()
    ax.scatter(x, y, s=200, c=colors, alpha=0.8, lw=0)
    if (predictPoint[0] != -1):
        ax.scatter(predictPoint[0], predictPoint[1], s=200, c='blue', alpha=1, lw=0)

    # ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color=[0, 0, 1], width=0.003)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.draw()
    plt.savefig(fileNameToSave + '.jpg')
    if (show == 1):
        plt.show()


def draw_learning_model(centrPoint, X, fileNameToSave='draw_learning_model', show=1):
    plt.figure()
    ax = plt.gca()
    for i in xrange(len(X) / 3):
        x = X[3 * i]
        y = X[3 * i + 1]
        c = color([x, y], centrPoint)
        ax.scatter(x, y, s=50, c=c, alpha=1, lw=0)

    ax.scatter(centrPoint[0], centrPoint[1], s=200, c='red', alpha=1, lw=0)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.draw()
    plt.savefig(fileNameToSave + '.jpg')
    if (show == 1):
        plt.show()
