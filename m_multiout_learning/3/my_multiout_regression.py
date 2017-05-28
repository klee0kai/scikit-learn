from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import shuffle
import my_model_multi_out as mModel

print 'load dataset.....'

# dataset = mModel.load_datasets(n=100000, pointsInFrame=20)
dataset = mModel.load_dataset_uniform(n=200, step=0.1)

# Split the data into training/testing sets
dataset_X_train = dataset[0][:-10]
dataset_X_test = dataset[0][-10:]

# Split the targets into training/testing sets
dataset_y_train = dataset[1][:-10]
dataset_y_test = dataset[1][-10:]

X = [[1, 23, 5], [1, 23, 8]]
Y = [[12, 6], [3, 9]]

mrg = MultiOutputRegressor(GradientBoostingRegressor(max_depth=10,
                                                     learning_rate=.02, min_samples_leaf=5,
                                                     min_samples_split=5))

for i in xrange(200):
    dataset_X_train, dataset_y_train = shuffle(dataset_X_train, dataset_y_train)
    print 'training ' + str(i) + ' ..... '
    mrg.fit(dataset_X_train, dataset_y_train)

# print 'training 2 ..... '
# mrg.fit(dataset_X_train, dataset_y_train)
# print 'training 3.....'
# mrg.fit(dataset_X_train, dataset_y_train)

print mrg.predict(dataset_X_test)

print 'testing.....'
pedictY = mrg.predict(dataset_X_test)

for i in xrange(len(pedictY)):
    y = dataset_y_test[i]
    x = dataset_X_test[i]
    predictY = mrg.predict(dataset_X_test[i])[0]
    mModel.draw_model(y, predictPoint=predictY, fileNameToSave='test step=0.1 i=' + str(i))
