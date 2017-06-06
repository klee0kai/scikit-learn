import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from math import *


# public
def load_dataset_uniform(n=20, step=0.01, maxPics=1, scaleByFrame=1., boxBorders=[0, 1], dim=2):
    x = []
    y = []
    for i in xrange(n):
        counts = random.randint(1, maxPics + 1)
        xFrame = []
        yFrame = [0] * (maxPics * dim)
        for j in xrange(1, counts + 1):
            centrP = list(random.rand(dim))
            xFrame = load_dataset_uniform_frame(step=step, centrPoint=centrP, lastDataFrame=xFrame,
                                                boxBorders=boxBorders)
            for k in xrange(dim):
                yFrame[2 * (j - 1) + k] = centrP[k]
            if (j == 1):
                for k in xrange(len(yFrame) / dim):
                    for g in xrange(dim):
                        yFrame[dim * k + g] = centrP[g]
        for k in xrange(len(yFrame) / dim):
            for j in xrange(k, len(yFrame) / dim):
                if (yFrame[dim * k] > yFrame[dim * j] or yFrame[dim * k] == yFrame[dim * j] and yFrame[dim * k + 1] >
                    yFrame[
                                2 * j + 1]):
                    G = list(random.rand(dim))
                    for g in xrange(dim):
                        G[g] = yFrame[dim * k + g]
                    for g in xrange(dim):
                        yFrame[dim * k + g] = yFrame[dim * j + g]
                    for g in xrange(dim):
                        yFrame[dim * j + g] = G[g]

        x = x + [xFrame]
        y = y + [yFrame]
    return [x, y]


# private
def load_dataset_uniform_frame(step=0.01, centrPoint=[0.5, 0.5], lastDataFrame=[], boxBorders=[0, 1]):
    box = np.arange(boxBorders[0], boxBorders[1], step)

    dim = len(centrPoint)

    points = [list(box[:])]
    for i in xrange(dim - 1):
        points = getDimUp(points, box)
    transporatedPoints = transporatePoints(points)

    lH = []
    if (len(lastDataFrame) > 0):
        lH = lastDataFrame[-len(lastDataFrame) / 3:]
    h = []
    for i in xrange(len(transporatedPoints)):
        h = h + [countingH(transporatedPoints[i], centrPoint)]

    if (len(lH) > 0):
        h = np.array(h)
        lH = np.array(lH)
        h = h + lH
        # normalize
        scale = 1. / max(h)
        h = h * scale
        h = list(h)

    outPoints = []
    for i in xrange(len(points)):
        outPoints = outPoints + points[i]
    return outPoints + h


# private
def getDimUp(dataset, box):
    count = len(dataset)
    size = len(dataset[0])
    sizeBox = len(box)

    newDataset = []
    for i in xrange(count):
        l = []
        for j in xrange(sizeBox):
            l = l + dataset[i]
        newDataset.append(l)

    l = []
    for i in xrange(sizeBox):
        for j in xrange(size):
            l.append(box[i])

    newDataset.append(l)
    return newDataset


# private
def transporatePoints(dataset):
    points = []
    for j in xrange(len(dataset[0])):
        point = []
        for i in xrange(len(dataset)):
            point.append(dataset[i][j])
        points.append(point)
    return points


# private
def color(point, centrPoint):
    for i in xrange(len(point)):
        s = (point[i] - centrPoint[i]) ** 2

    d = sqrt(s)
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
    for i in xrange(len(point)):
        s = (point[i] - centrPoint[i]) ** 2
    d = sqrt(s)
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
