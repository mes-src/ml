import mglearn
from matplotlib import pyplot as plt 

waves = mglearn.plots.plot_linear_regression_wave()
print(waves)
plt.show()

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
x,y = mglearn.datasets.make_wave(n_samples = 60)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state =42)
lr = LinearRegression().fit(x_train, y_train)
print(lr.coef_)
print(lr.intercept_)


print(lr.score(x_train, y_train))
print(lr.score(x_test, y_test))