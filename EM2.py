import pandas as pd
import seaborn as sns; sns.set()
from sklearn.datasets import make_classification
import random
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import time
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.mixture import GaussianMixture

from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import calinski_harabasz_score
from sklearn.decomposition import PCA


import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import completeness_score
from sklearn.metrics import homogeneity_score
from sklearn import mixture

# color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
#                               'darkorange'])
color_iter = itertools.cycle(['navy', 'darkorange', 'green', 'gold',
                              'darkorange', 'red', 'brown', 'purple', 'yellow', 'pink', 'grey'])


#from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py
def plot_results(X, y, n, title):

    X = PCA(n_components=2).fit_transform(X)
    gmm = GaussianMixture(n_components=n, covariance_type='full').fit(X)
    Y_ = gmm.predict(X)
    means = gmm.means_
    covariances = gmm.covariances_


    # splot = plt.subplot(1, 1, 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
    splot = ax1
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        ax1.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 15, color=color, label = 'Cluster ' + str(i))

        # # Plot an ellipse to show the Gaussian component
        # angle = np.arctan(u[1] / u[0])
        # angle = 180. * angle / np.pi  # convert to degrees
        # ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        # ell.set_clip_box(splot.bbox)
        # ell.set_alpha(0.5)
        # splot.add_artist(ell)


    plt.xticks(())
    plt.yticks(())
    ax1.legend(prop={'size': 15})
    ax1.set_title(title)

    ones = np.where(y == 1)
    x1 = X[ones]

    ax2.scatter(x1[:, 0], x1[:, 1], 15, color='red', label = 'Class 1')

    zeros = np.where(y == 0)
    x0 = X[zeros]
    ax2.scatter(x0[:, 0], x0[:, 1], 15, color='blue', label = 'Class 2')

    # ax2.set_xticks([-1.0, -.5, 0, 1, 1.5])
    # ax2.set_yticks([-.5,0,.5,1])
    ax2.set_title("Labels")
    ax2.legend(prop={'size': 15})


    plt.savefig("B-EM-Plot-" + str(n) + "-clust.png")




def main():


    datafile = "Data/biodeg.csv"
    data = np.genfromtxt(datafile, delimiter=";")

    data = data[:910, :]
    data = np.delete(data, slice(650, 820), 0)
    np.random.shuffle(data)

    X = data[:, :-1]
    y = data[:, -1]

    # from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, train_size=.8)

    # from https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data
    sc = MinMaxScaler()
    train_x = sc.fit_transform(train_x)
    test_x = sc.transform(test_x)

    X = train_x


    silhouette_scores = []
    chs = []
    r = []
    c = []
    h = []

    N = range(2, 10)
    for i in N:
        em = GaussianMixture(i, covariance_type='full',random_state=42)
        labels = em.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
        chs.append(calinski_harabasz_score(X, labels))
        r.append(adjusted_rand_score(labels, train_y))
        c.append(completeness_score(labels, train_y))
        h.append(homogeneity_score(labels, train_y))

    print("Rand Score")
    print("2 clusters: " + str(r[0]))
    print("3 clusters: " + str(r[1]))

    print("Homo Score")
    print("2 clusters: " + str(h[0]))
    print("3 clusters: " + str(h[1]))

    print("Compeletness Score")
    print("2 clusters: " + str(c[0]))
    print("3 clusters: " + str(c[1]))

    print("CHS Score")
    print("2 clusters: " + str(chs[0]))
    print("3 clusters: " + str(chs[1]))

    print("Silhouette Score")
    print("2 clusters: " + str(silhouette_scores[0]))
    print("3 clusters: " + str(silhouette_scores[1]))

    plt.plot(N, chs, marker='o')
    plt.title('Calinski Harabasz Score')
    plt.xlabel('Number of clusters')
    plt.ylabel('CHS')
    plt.savefig('B-EM-CHS.png')
    plt.close()
    plt.figure()


    # from https://medium.com/mlearning-ai/k-means-clustering-with-scikit-learn-e2af706450e4
    plt.plot(N, silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette score')
    plt.title("Silhouette Score")
    plt.savefig('B-EM-Silhouette.png')
    plt.close()
    plt.figure()

    plot_results(X, train_y, 2, 'EM K=2')


if __name__ == "__main__":
    np.random.seed(26)
    random.seed(42)
    main()

