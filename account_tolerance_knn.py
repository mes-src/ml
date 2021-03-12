import fmr_init as fmri
import pandas as pd
from pandas.plotting import scatter_matrix
import mglearn

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


X = fmri.get_data() #The df of data contianing all feature set

ptarget = 'C:/Users/MichaelSands/Desktop/account_list_with_manual_classifications.xlsx'
tg = pd.read_excel(ptarget)
tg['target_class'] = [item for item in range(len(tg))]
tg.loc[tg.HumanLabeledClassification == 'Aggressive', 'target_class'] = 0
tg.loc[tg.HumanLabeledClassification == 'Moderate/Aggressive', 'target_class'] = 1
tg.loc[tg.HumanLabeledClassification == 'Moderate', 'target_class'] = 2
tg.loc[tg.HumanLabeledClassification == 'Conservative', 'target_class'] = 3
tg = tg.set_index('acctno')

print(tg)
target_class = fmri.get_target_class_list()

y_train = fmri.get_target_labels_list()
y_train = tg['target_class'].to_list() #'[] #target labels #The custom target labels from the training set (pre labeled) # a one dimensional list of class
print(y_train)
# train test split
#X_train, X_test, y_train,y_test = train_test_split(X) #TargetLabels#



labeled_df = X.merge(tg, left_index = True, right_index = True)
labeled_df = labeled_df[['age_x','cashpct_x','equitypct_x','fipct_x','goldpct_x','target_class']]
labeled_df = labeled_df.rename(columns = {'age_x':'age','equitypct_x':'equitypct','cashpct_x':'cashpct','fipct_x':'fipct','goldpct_x':'goldpct'})
unlabeled_df = labeled_df[['age','cashpct','equitypct','fipct','goldpct']]
print(labeled_df)
print(labeled_df.columns)

# grr = scatter_matrix(unlabeled_df, c=labeled_df.target_class.to_list(), figsize=(10,5), marker = 'o', hist_kwds={'bins':20}, s=5, alpha =0.8,cmap=mglearn.cm3)#figsize width, height
# plt.legend(loc="upper left")
# plt.show()



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
X = unlabeled_df
knn.fit(X, y_train)
KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski', metric_params= None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')
print(X)


import seaborn as sns
g = sns.pairplot(labeled_df, hue ='target_class',markers=["o", "s", "D",'+','2','1'] )
g.fig.set_figwidth(12)
g.fig.set_figheight(6)
plt.show()


new = np.array([[55,.0,.60,.00,.05]]) # Aggressive
# new = np.array([[75,.0,.60,.00,.05]]) # Moderate




prediction = knn.predict(new)[0]
print('	age	 cashpct	equitypct	fipct	goldpct')
print(new)
print(target_class)
print('PREDICTION: ' + str(target_class[prediction]))