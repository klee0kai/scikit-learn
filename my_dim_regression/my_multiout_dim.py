import os
import cPickle
import datetime

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor, \
    IsolationForest, RandomForestRegressor
from sklearn.utils import shuffle
from numpy import array
import my_model_dim as mModel
import my_log_errors as errorLog

folder = './testDim'
folderPostFix = 0
while (os.path.exists(folder + str(folderPostFix))):
    folderPostFix = folderPostFix + 1
folder = folder + str(folderPostFix) + '/'

print 'results in ' + folder

print 'load dataset.....'

# dataset = mModel.load_datasets(n=100000, pointsInFrame=20)
datasetSize = 5000
dataSetStep = 0.11
boxBorders = [0, 1]
maxPics = 1
dim = 3
dataset = mModel.load_dataset_uniform(n=datasetSize, step=dataSetStep, maxPics=maxPics, scaleByFrame=1,
                                      boxBorders=boxBorders, dim=dim)

# Split the data into training/testing sets
dataset_X_train = dataset[0][:-25]
dataset_X_test = dataset[0][-25:]

# Split the targets into training/testing sets
dataset_y_train = dataset[1][:-25]
dataset_y_test = dataset[1][-25:]

# load it again
# with open(folder + 'mrg.pkl', 'rb') as fid:
#     mrg = cPickle.load(fid)


mrg = MultiOutputRegressor(GradientBoostingRegressor(max_depth=10, learning_rate=0.1))

startTime = datetime.datetime.now()

# dataset_X_train, dataset_y_train = shuffle(dataset_X_train, dataset_y_train)
print 'training ..... '
mrg.fit(dataset_X_train, dataset_y_train)
# count errors
errorLog.addErrorLog(mrg.predict(dataset_X_test) - array(dataset_y_test), 1)

endTime = datetime.datetime.now()
elapsedTime = endTime - startTime
print 'spend time :' + str(elapsedTime)

print 'save neural network to disk.....'

if not os.path.exists(folder):
    os.makedirs(folder)

# save the classifier
with open(folder + 'mrg.pkl', 'wb') as fid:
    cPickle.dump(mrg, fid)

print 'testing.....'
predictYs = mrg.predict(dataset_X_test)
predictYs = list(predictYs)
print str(predictYs)

print 'print to log file.....'
errorLog.saveLog(folder)
errorLog.draw(folder + "error_function_", show=0)

f1 = open(folder + 'log.txt', 'w+')
f1.write(str(mrg) + '\n' + '\n')
f1.write('datasetSize= ' + str(datasetSize) + '\n')
f1.write('dataSetStep= ' + str(dataSetStep) + '\n')
f1.write('boxBorders= ' + str(boxBorders) + '\n')
f1.write('maxPics= ' + str(maxPics) + '\n')
f1.write('dim = ' + str(dim))
f1.write('spended time: ' + str(elapsedTime) + '\n\n')

for i in xrange(len(predictYs)):
    y = dataset_y_test[i]
    x = dataset_X_test[i]
    predictY = predictYs[i]
    f1.write('test ' + str(i) + '\n')
    f1.write('centrPoint= ' + str(y) + '\n')
    f1.write('predictCentrPoint ' + str(predictY) + '\n')
    f1.write('err= ' + str(array(y) - array(predictY)) + '\n' + '\n')

f1.close()
print 'finish.....'
print 'results in ' + folder
