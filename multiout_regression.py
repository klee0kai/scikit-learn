from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_regression(n_samples=10, n_targets=3, random_state=1)
print MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(X, y).predict(X)
