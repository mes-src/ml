from matplotlib import pyplot as plt 
import mglearn
import numpy as np 

def forge():
	x,y = mglearn.datasets.make_forge()
	mglearn.discrete_scatter(x[:, 0 ], x[:,1], y)
	plt.legend(['Class 0', 'Class 1'], loc = 4)
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	print("x.shape: {}".format(x.shape))
	plt.show()

def cancer():
	from sklearn.datasets import load_breast_cancer
	cancer = load_breast_cancer()

	print("cancer.keys" + '    ' + str(cancer.keys()))
	print("cancer.data.shape" + '   ' + str(cancer.data.shape))
	countsPerClass = {n: v for n,v in zip(cancer.target_names, np.bincount(cancer.target))}
	print(countsPerClass)
	print("cancer.feature_names" + '   ' + str(cancer.feature_names))


cancer()