import os
import cPickle
import datetime

import my_model_NS_in_CS_dim as mModel

folder = './testSearchInCS_NewModelRating'
folderPostFix = 0
while (os.path.exists(folder + str(folderPostFix))):
    folderPostFix = folderPostFix + 1
folder = folder + str(folderPostFix) + '/'

print 'results in ' + folder

# mrg specifications
dataSetStep = 0.11
boxBorders = [0, 1]
# load mrg again
with open('mrg' + '.pkl', 'rb') as fid:
    mrg = cPickle.load(fid)
# --------------


startTime = datetime.datetime.now()

borderSize = 20
predictResult = [0, 0, 0, 0]
denormaliseResult = [0, 0, 0, 0]
rating = 0

predictResults = []
denormaliseResults = []
ratings = []

rating = mModel.countingH(denormaliseResult)
ratings.append(rating)
denormaliseResults.append(denormaliseResult)
predictResults.append(predictResult)

while (borderSize > 0.1):
    border = mModel.getBorder(predictResult=predictResult, borderSize=borderSize)
    X = mModel.load_dataset_uniform_frame(border=border, boxBorders=boxBorders, dataSetStep=dataSetStep)
    predictResult = mrg.predict(X)[0]
    denormaliseResult = mModel.denormalisePoint(point=predictResult, border=border)
    rating = mModel.countingH(denormaliseResult)
    borderSize = borderSize * 0.70
    ratings.append(rating)
    denormaliseResults.append(denormaliseResult)
    predictResults.append(predictResult)

endTime = datetime.datetime.now()
elapsedTime = endTime - startTime
print 'spend time :' + str(elapsedTime)

print 'save neural network to disk.....'

if not os.path.exists(folder):
    os.makedirs(folder)

print 'print to log file.....'

f1 = open(folder + 'log.txt', 'w+')
f1.write(str(mrg) + '\n' + '\n')
f1.write('mrg specifications: \n')
f1.write('dataSetStep = ' + str(dataSetStep) + '\n')
f1.write('boxBorders = ' + str(boxBorders) + '\n'+'\n')

f1.write('predictResult = ' + str(predictResult) + '\n')
f1.write('denormaliseResult = ' + str(denormaliseResult) + '\n')
f1.write('rating = ' + str(rating) + '\n')
f1.write('spended time: ' + str(elapsedTime) + '\n\n')

for i in xrange(len(denormaliseResults)):
    predictResult = predictResults[i]
    denormaliseResult = denormaliseResults[i]
    rating = ratings[i]
    f1.write('predictResult = ' + str(predictResult) + '\n')
    f1.write('denormaliseResult = ' + str(denormaliseResult) + '\n')
    f1.write('rating = ' + str(rating) + '\n')
    f1.write('\n')
f1.close()

print 'finish.....'
print 'results in ' + folder
