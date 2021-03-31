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
from scipy.spatial.distance import cdist

from sklearn.metrics import silhouette_score

from yellowbrick.cluster import SilhouetteVisualizer

from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import numpy as np
from gap_statistic import OptimalK
from sklearn.linear_model import LogisticRegression
from kneed import DataGenerator, KneeLocator
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
from yellowbrick.features import Manifold
import umap



#from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
def plotit(X, y, n):
    reduced_data = PCA(n_components=2).fit_transform(X)
    kmeans = KMeans(init="k-means++", n_clusters=n, n_init=4)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation="nearest",
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.tab10, aspect="auto", origin="lower", )
               # cmap=plt.cm.Paired, aspect="auto", origin="lower", )

    ones = np.where(y == 1)
    x1 = reduced_data[ones]

    plt.plot(x1[:, 0], x1[:, 1], 'k.', color = 'red', markersize=5, label = 'Class 1')

    zeros = np.where(y == 0)
    x0 = reduced_data[zeros]
    plt.plot(x0[:, 0], x0[:, 1], 'k.', color='blue', markersize=5, label = 'Class 2')


    # plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)



    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
                color="w", zorder=10)
    plt.title("K-means Clustering K=" + str(n))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(prop={'size': 15})
    plt.xticks(())
    plt.yticks(())
    plt.savefig("S-KM-" + str(n) + "-clust.png")



def main():


    datafile = "Data/spambase.data"
    data = np.genfromtxt(datafile, delimiter=",")
    data = data[:-975, :]
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]
    R = X


    # from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, train_size=.8)

    # from https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data
    sc = MinMaxScaler()
    train_x = sc.fit_transform(train_x)
    test_x = sc.transform(test_x)

    X = train_x
    y = train_y


    #From https://medium.com/pursuitnotes/k-means-clustering-model-in-6-steps-with-python-35b532cfa8ad
    wcss = []
    silhouette_scores = []
    chs = []
    h = []
    c = []
    r = []
    N = range(2,11)
    for i in N:
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)

        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        chs.append(calinski_harabasz_score(X, kmeans.labels_))
        h.append(homogeneity_score(y, kmeans.labels_))
        c.append((completeness_score(y, kmeans.labels_)))
        r.append(adjusted_rand_score(y, kmeans.labels_))


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
    plt.savefig('S-KM-CHS.png')
    plt.close()
    plt.figure()


    # from https://medium.com/mlearning-ai/k-means-clustering-with-scikit-learn-e2af706450e4
    plt.plot(N, silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette Score')
    plt.savefig('S-KM-Silhouette.png')
    plt.close()
    plt.figure()


    plotit(X,y,2)
    plotit(X,y,3)









if __name__ == "__main__":
    np.random.seed(26)
    random.seed(42)
    main()

