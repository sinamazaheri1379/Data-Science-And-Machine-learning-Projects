import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
X = np.genfromtxt("Iris.csv", delimiter=',')
X_test = X[1:, 1:X.shape[1] - 1].copy().transpose()
X_final = (X_test - np.mean(X_test, axis=1).reshape((X_test.shape[0], 1)).dot(np.ones((1, X_test.shape[1])))) / (np.std(X_test, axis=1).reshape((X_test.shape[0], 1)).dot(np.ones((1, X_test.shape[1]))))
basis = np.linalg.eig(1 / 149 * X_final.dot(X_final.transpose()))[1][0:, 0:2].copy()
final_data_set = basis.transpose().dot(X_final)
group0 = np.full((50,), 1)
group1 = np.full((50,), 2)
group2 = np.full((50,), 3)
final_group = np.hstack((group0, group1, group2))
classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
map_color = ListedColormap(['r', 'g', 'b'])
scatter = plt.scatter(final_data_set[0, 0:], final_data_set[1, 0:], c=final_group, cmap=map_color)
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.show()