import os
import cPickle
import datetime

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor, \
    IsolationForest, RandomForestRegressor
from sklearn.utils import shuffle
from numpy import array
import my_model_NS_in_CS as mModel
import my_log_errors as errorLog

folder = './NSModelMap'
folderPostFix = 0
while (os.path.exists(folder + str(folderPostFix)) ):
    folderPostFix = folderPostFix + 1
folder = folder + str(folderPostFix) + '/'

print 'results in ' + folder

if not os.path.exists(folder):
    os.makedirs(folder)

mModel.draw_model(fileNameToSave=folder + 'map', show=0)

print 'finish....'
