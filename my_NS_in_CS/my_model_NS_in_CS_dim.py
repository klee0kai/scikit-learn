import numpy as np
from math import *
import model_pendulum_on_carriage as mPendulum


# public
def load_dataset_uniform_frame(border=[0, 1, 0, 1, 0, 1, 0, 1], boxBorders=[0, 1], dataSetStep=0.99):
    dim = 4

    box = np.arange(boxBorders[0], boxBorders[1], dataSetStep)
    points = [list(box[:])]
    for i in xrange(dim - 1):
        points = getDimUp(points, box)
    transporatedPoints = transporatePoints(points)

    h = []
    for i in xrange(len(transporatedPoints)):
        # denormalise points to
        point = denormalisePoint(transporatedPoints[i], border=border)
        h = h + [countingH(point)]

    outPoints = []
    for i in xrange(len(points)):
        outPoints = outPoints + points[i]
    return outPoints + h


    # prepare border for denormalise


def getBorder(predictResult=[0, 0, 0, 0], borderSize=10):
    border = []
    for predictP in predictResult:
        border = border + [predictP - borderSize / 2., predictP + borderSize / 2.]
    return border


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


def denormalise(a, min, max):
    l = max - min
    a = a * l + min
    return a


def denormalisePoint(point, border):
    for i in xrange(len(point)):
        point[i] = denormalise(point[i], border[2 * i], border[2 * i + 1])
    return point


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
def color(point):
    red = countingH(point)
    blue = (1. - sqrt(red)) / 5.
    if (red < 0):
        red = 0
    if (blue > 1):
        blue = 1
    if (blue < 0):
        blue = 0
    return [red, 0, blue]


# private
def countingH(point):
    return mPendulum.countingH(point)


# private
def countingHs(points):
    return mPendulum.countingHs(points)
