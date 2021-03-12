import pandas as pd 
import numpy as np 

from matplotlib import pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import mglearn 
x,y = mglearn.datasets.make_forge()
fig,axes = plt.subplots(1,2,figsize=(10,3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
	clf = model.fit(x,y)
	mglearn.plots.plot_2d_separator(clf,x,fill=False,eps=0.5,ax=ax, alpha = .7)
	mglearn.discrete_scatter(x[:,0], x[:,1], y, ax=ax)
	ax.set_title(clf.__class__.__name__)

plt.show()