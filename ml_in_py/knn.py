import mglearn
from matplotlib import pyplot as plt 

mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.show()

from sklearn.model_selection import train_test_split
x,y = mglearn.datasets.make_forge()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(x_train, y_train)
print(clf.predict(x_test))
print('Accuracy Score: ')
print(clf.score(x_test, y_test))


#Ploting 1,3,9 n neighbors for comparison
fig, axes = plt.subplots(1,3, figsize=(10,3))
for n_neighbors, ax in zip([1,3,9], axes):
	clf = KNeighborsClassifier(n_neighbors).fit(x, y)
	mglearn.discrete_scatter(x[:,0], x[:,1], y, ax=ax)
	ax.set_title(n_neighbors)
	ax.set_xlabel('feature 0')
	ax.set_ylabel('feature 1')
axes[0].legend(loc =3)
plt.show()