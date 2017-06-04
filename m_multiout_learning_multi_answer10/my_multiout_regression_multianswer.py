import os
import cPickle
import datetime

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor, \
    IsolationForest, RandomForestRegressor
from sklearn.utils import shuffle
from numpy import array
import my_model_multi_out_multianswer as mModel
import my_log_errors as errorLog

folder = './testMultiAnswer'
folderPostFix = 0
while (os.path.exists(folder + str(folderPostFix) + '/')):
    folderPostFix = folderPostFix + 1
folder = folder + str(folderPostFix) + '/'

if not os.path.exists(folder):
    os.makedirs(folder)

print 'results in ' + folder

print 'load dataset.....'

# dataset = mModel.load_datasets(n=100000, pointsInFrame=20)
datasetSize = 100
dataSetStep = 0.197
boxBorders = [0.1, 0.9]
maxPics = 2
dataset = mModel.load_dataset_uniform(n=datasetSize, step=dataSetStep, maxPics=maxPics, scaleByFrame=0.8,
                                      boxBorders=boxBorders)

# Split the data into training/testing sets
dataset_X_train = dataset[0][:-25]
dataset_X_test = dataset[0][-25:]

# Split the targets into training/testing sets
dataset_y_train = dataset[1][:-25]
dataset_y_test = dataset[1][-25:]

# load it again
# with open(folder + 'mrg.pkl', 'rb') as fid:
#     mrg = cPickle.load(fid)


mrg = MultiOutputRegressor(GradientBoostingRegressor(max_depth=10, learning_rate=0.03))

startTime = datetime.datetime.now()

learning_cicles = 100
for i in xrange(learning_cicles):
    dataset_X_train, dataset_y_train = shuffle(dataset_X_train, dataset_y_train)
    print 'training ' + str(i) + ' left: ' + str(learning_cicles - i) + ' ..... '
    mrg.fit(dataset_X_train, dataset_y_train)

    # count errors
    errorLog.addErrorLog(mrg.predict(dataset_X_test) - array(dataset_y_test), (i+1) * datasetSize)

endTime = datetime.datetime.now()
elapsedTime = endTime - startTime
print 'spend time :' + str(elapsedTime)

print 'save neural network to disk.....'

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
f1.write('learning_cicles= ' + str(learning_cicles) + '\n\n')
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
print 'drawing.....'

for i in xrange(len(dataset_X_test)):
    y = dataset_y_test[i]
    x = dataset_X_test[i]
    mModel.draw_model(y, predictPoints=predictYs[i], fileNameToSave=folder + 'test_' + str(i), show=0)
    mModel.draw_training_model(x, y, fileNameToSave=folder + 'training_model_' + str(i), show=0)

print 'finish.....'
print 'results in ' + folder
