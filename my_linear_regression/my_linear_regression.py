import matplotlib.pyplot as plt
import numpy as np
import my_model as mModel
from sklearn import linear_model

# Load the diabetes dataset
print 'load dataset.....'
dataset = mModel.load_datasets()


# Use only one feature
diabetes_X = dataset[0]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-1]
diabetes_X_test = diabetes_X[-1:]

# Split the targets into training/testing sets
diabetes_y_train = dataset[1][:-1]
diabetes_y_test = dataset[1][-1:]

print 'training.....'
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

print 'test ...'
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))


# Plot outputs
plt.scatter(diabetes_X_test[:-1], diabetes_y_test,  color='black')
plt.plot(diabetes_X_test[:-1], regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()