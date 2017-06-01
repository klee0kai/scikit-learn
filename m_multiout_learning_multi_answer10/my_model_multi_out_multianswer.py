import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from math import *


# public
def draw_model(centrPoints, predictPoints=[], fileNameToSave="", show=1):
    box = np.arange(0, 1, 0.01)

    y = []
    x = []
    for i in box:
        y = y + list(np.linspace(i, i, len(box)))
        x = x + list(box)
    h = np.array(0)
    for centrPoint in list(centrPoints[2 * i:i * 2 + 2] for i in xrange(len(centrPoints) / 2)):
        h = h + np.array(countingHs(x + y, centrPoint))
    h = h * 1. / max(h)
    h = list(h)

    colors = []
    for i in xrange(len(h)):
        red = h[i]
        blue = (1. - sqrt(red)) / 5.
        if (red < 0):
            red = 0
        if (blue > 1):
            blue = 1
        if (blue < 0):
            blue = 0
        colors.append([red, 0, blue])

    plt.figure()
    ax = plt.gca()
    ax.scatter(x, y, s=200, c=colors, alpha=0.8, lw=0)
    if predictPoints != []:
        for predictPoint in list(predictPoints[2 * i:i * 2 + 2] for i in xrange(len(predictPoints) / 2)):
            ax.scatter(predictPoint[0], predictPoint[1], s=200, c='blue', alpha=1, lw=0)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.draw()
    if (fileNameToSave != ""):
        plt.savefig(fileNameToSave + '.jpg')
    if (show == 1):
        plt.show()


# public
def draw_training_model(X, Y, show=1, fileNameToSave=""):
    blockSize = len(X) / 3
    x = X[0:blockSize]
    y = X[blockSize:2 * blockSize]
    h = X[2 * blockSize:3 * blockSize]

    colors = []
    for i in xrange(len(x)):
        red = h[i]
        blue = (1. - sqrt(red)) / 5.
        if (red < 0):
            red = 0
        if (blue > 1):
            blue = 1
        if (blue < 0):
            blue = 0
        colors = colors + [[red, 0, blue]]

    # prepare figure
    plt.figure()
    ax = plt.gca()
    ax.scatter(x, y, s=50, c=colors, alpha=1, lw=0)

    centrPoints = list(Y[2 * i:i * 2 + 2] for i in xrange(len(Y) / 2))
    for centrP in centrPoints:
        ax.scatter(centrP[0], centrP[1], s=200, c='red', alpha=1, lw=0)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.draw()
    # save
    if (fileNameToSave != ""):
        plt.savefig(fileNameToSave + '.jpg')
    if (show == 1):
        plt.show()


# public
def load_datasets(n=20, pointsInFrame=100, maxPics=1, scaleByFrame=1.):
    x = []
    y = []
    for i in xrange(n):
        counts = random.randint(1, maxPics + 1)
        xFrame = np.array(0)
        yFrame = [0] * (maxPics * 2)
        for j in xrange(1, counts + 1):
            centrP = list(random.rand(2))
            data_frame_X = load_data_frame_X(countPoints=pointsInFrame, centrPoint=centrP, scale=scaleByFrame ** j)
            xFrame = xFrame + np.array(data_frame_X)
            yFrame[2 * (j - 1)] = centrP[0]
            yFrame[2 * (j - 1) + 1] = centrP[1]
            if (j == 1):
                for k in xrange(len(yFrame) / 2):
                    yFrame[2 * k] = centrP[0]
                    yFrame[2 * k + 1] = centrP[1]

        x = x + list(xFrame)
        y = y + list(yFrame)
    return [x, y]


# public
def load_dataset_uniform(n=20, step=0.01, maxPics=1, scaleByFrame=1.):
    x = []
    y = []
    for i in xrange(n):
        counts = random.randint(1, maxPics + 1)
        xFrame = []
        yFrame = [0] * (maxPics * 2)
        for j in xrange(1, counts + 1):
            centrP = list(random.rand(2))
            xFrame = load_dataset_uniform_frame(step=step, centrPoint=centrP, lastDataFrame=xFrame)
            yFrame[2 * (j - 1)] = centrP[0]
            yFrame[2 * (j - 1) + 1] = centrP[1]
            if (j == 1):
                for k in xrange(len(yFrame) / 2):
                    yFrame[2 * k] = centrP[0]
                    yFrame[2 * k + 1] = centrP[1]
        x = x + [xFrame]
        y = y + [yFrame]
    return [x, y]


# private
def load_dataset_uniform_frame(step=0.01, centrPoint=[0.5, 0.5], lastDataFrame=[]):
    box = np.arange(0, 1, step)

    y = []
    x = []
    for i in box:
        y = y + list(np.linspace(i, i, len(box)))
        x = x + list(box)

    lH = []
    if (len(lastDataFrame) > 0):
        lH = lastDataFrame[-len(lastDataFrame) / 3:]
    h = []
    for i in xrange(len(x)):
        h = h + [countingH([x[i], y[i]], centrPoint)]

    if (len(lH) > 0):
        h = np.array(h)
        lH = np.array(lH)
        h = h + lH
        # normalize
        scale = 1. / max(h)
        h = h * scale
        h = list(h)

    return x + y + h


# private
def load_data_frame_X(countPoints=3, centPoint=list(random.rand(2)), scale=1.):
    x = list(random.rand(countPoints))
    y = list(random.rand(countPoints))
    h = []
    for i in xrange(countPoints):
        h = h + [countingH([x[i], y[i]], centPoint) * scale]
    return [x + y + h]


# private
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


# private
def color(point, centPoint):
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


# private
def countingH(point, centrPoint):
    d = sqrt((point[0] - centrPoint[0]) ** 2 + (point[1] - centrPoint[1]) ** 2)
    h = (1 - d) ** 2
    return h


# private
def countingHs(points, centrPoint):
    Xs = points[:len(points) / 2]
    Ys = points[len(points) / 2:len(points)]
    hs = []
    for i in xrange(len(Xs)):
        d = sqrt((Xs[i] - centrPoint[0]) ** 2 + (Ys[i] - centrPoint[1]) ** 2)
        h = [(1 - d) ** 2]
        hs = hs + h
    return hs
