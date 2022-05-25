from sklearn import svm, datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn import metrics
from sklearn.model_selection import train_test_split
from os import listdir
import skimage
#
for i, C in [(1, 300), (2, 400), (3, 1000), (4, 320), (5, 700)]:
    X, y = make_blobs(n_samples=100 * (i + 1), centers=2, random_state=3)
    clf = svm.SVC(kernel="linear", C=C)
    clf.fit(X, y)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    ax.contour(
        XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
    )
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    plt.show()
##
for tup in ['rbf','poly','sigmoid']:
    xx, yy = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))
    np.random.seed(0)
    X = np.random.randn(300, 2)
    Y = np.logical_or(X[:, 0] > 0, X[:, 1] > 0)
    clf = svm.NuSVC(kernel=tup, gamma='auto', coef0=0.04, degree=2)
    clf.fit(X, Y)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.PuOr_r,
    )
    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles="dashed")
    plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired, edgecolors="k")
    plt.xticks(())
    plt.yticks(())
    plt.axis([-3, 3, -3, 3])
    plt.show()
#
for tup in ['rbf','poly','sigmoid']:
    xx, yy = np.meshgrid(np.linspace(-10, 10, 500), np.linspace(-10, 10, 500))
    np.random.seed(0)
    X = np.random.randn(300, 2)
    Y = np.logical_not(X[:, 0] < 0)
    clf = svm.NuSVC(kernel=tup, gamma='auto', coef0=0.04, degree=2)
    clf.fit(X, Y)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.PuOr_r,
    )
    plt.legend([tup])
    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles="dashed")
    plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired, edgecolors="k")
    plt.xticks(())
    plt.yticks(())
    plt.axis([-3, 3, -3, 3])
    plt.show()


digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
clf = svm.NuSVC(kernel='rbf', gamma=0.001)
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=True
)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
##

X = []
y = []

mypath = './persian_LPR/'
class_list = ['2', '3', '7', 'S', 'W']
class_dict = {'2': 0, '3': 1, '7': 2, 'S': 3, 'W': 4}
for classes in ['2', '3', '7', 'S', 'W']:
    final = mypath + classes
    for file in listdir(final):
        X.append(skimage.io.imread(final + '/' + file).tolist())
        y.append(class_dict[classes])


X = np.reshape(np.array(X), (1500, -1))
y = np.array(y)
clf = svm.NuSVC(kernel='poly')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True
)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print(
     f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n")