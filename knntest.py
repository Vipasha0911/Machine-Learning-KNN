import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import KNN

cmap =ListedColormap(['#FF0000','#00ff00','#0000ff'])

iris = datasets.load_iris()
x,y = iris.data, iris.target

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state= 1234)


clf = KNN(k=3)
clf.fit(x_train,y_train)
predictions = clf.predict(x_test)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)