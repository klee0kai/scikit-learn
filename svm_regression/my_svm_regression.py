from sklearn import svm
import my_model_svm as mModel

print 'load dataset.....'
dataset = mModel.load_datasets()


# Use only one feature
diabetes_X = dataset[0]

# Split the data into training/testing sets
dataset_X_train = diabetes_X[:-1]
dataset_X_test = diabetes_X[-1:]

# Split the targets into training/testing sets
dataset_y_train = dataset[1][:-1]
dataset_y_test = dataset[1][-1:]

print 'training.....'

clf = svm.SVR()
clf.fit(dataset_X_train, dataset_y_train)

print clf.predict(dataset_y_test)