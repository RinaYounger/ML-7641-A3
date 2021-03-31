from sklearn.decomposition import PCA
import numpy as np
from numpy.testing import assert_array_almost_equal
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import seaborn as sb
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from sklearn.random_projection import GaussianRandomProjection
from sklearn import utils
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from kneed import DataGenerator, KneeLocator
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import completeness_score
from sklearn.metrics import homogeneity_score


import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture




import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

# color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
#                               'darkorange'])
color_iter = itertools.cycle(['navy', 'darkorange', 'green', 'gold',
                              'darkorange', 'red', 'cornflowerblue', 'purple', 'gold', 'pink', 'grey'])


#from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py
def plotem(X, y, n, name):

    X = PCA(n_components=2).fit_transform(X)
    gmm = GaussianMixture(n_components=n, covariance_type='full').fit(X)
    Y_ = gmm.predict(X)
    means = gmm.means_
    covariances = gmm.covariances_


    splot = plt.subplot(1, 1, 1)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
    # splot = ax1
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
        splot.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 15, color=color, label = 'Cluster ' + str(i))

        # # Plot an ellipse to show the Gaussian component
        # angle = np.arctan(u[1] / u[0])
        # angle = 180. * angle / np.pi  # convert to degrees
        # ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        # ell.set_clip_box(splot.bbox)
        # ell.set_alpha(0.5)
        # splot.add_artist(ell)


    # plt.xticks(())
    # plt.yticks(())
    splot.legend(prop={'size': 15})
    splot.set_title("EM - " + name + " No. Clusters = " + str(n))

    # ones = np.where(y == 1)
    # x1 = X[ones]
    #
    # ax2.scatter(x1[:, 0], x1[:, 1], 15, color='red', label = 'Class 1')
    #
    # zeros = np.where(y == 0)
    # x0 = X[zeros]
    # ax2.scatter(x0[:, 0], x0[:, 1], 15, color='blue', label = 'Class 2')
    #
    # # ax2.set_xticks([-1.0, -.5, 0, 1, 1.5])
    # # ax2.set_yticks([-.5,0,.5,1])
    # ax2.set_title("Labels")
    # ax2.legend(prop={'size': 15})


    plt.savefig("S-EM-"+name+"-"+str(n)+"clust.png")
    plt.close()
    plt.figure()





#from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
def plotkm(X, y, n, name):
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
               cmap=plt.cm.tab10, aspect="auto", origin="lower")

    ones = np.where(y == 1)
    x1 = reduced_data[ones]

    plt.plot(x1[:, 0], x1[:, 1], 'k.', color = 'red', markersize=5, label = "Class 1")

    zeros = np.where(y == 0)
    x0 = reduced_data[zeros]
    plt.plot(x0[:, 0], x0[:, 1], 'k.', color='blue', markersize=5, label = "Class 2")


    # plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)



    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
                color="w", zorder=10)
    plt.title("K-Means - " + name + " No. Clusters = " + str(n))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(prop={'size': 15})
    plt.xticks(())
    plt.yticks(())
    plt.savefig("S-KM-"+name+"-"+str(n)+"clust.png")
    plt.close()
    plt.figure()




def getScores(R, y):

    pca = PCA(n_components=34, random_state=42)
    X = pca.fit_transform(R)
    n = 3

    kmeans = KMeans(random_state=42, n_clusters=n)
    kmeans.fit(X)
    print("PCA - KM - " + str(n) +" Clust")
    print("Homo" + str(homogeneity_score(y, kmeans.labels_)))
    print("Complete" + str((completeness_score(y, kmeans.labels_))))
    print("Rand" + str(adjusted_rand_score(y, kmeans.labels_)))
    print("Silhouette" + str(silhouette_score(X, kmeans.labels_)))
    print("Calinski Harabasz" + str(calinski_harabasz_score(X, kmeans.labels_)))

    gm = GaussianMixture(n, covariance_type='full',random_state=42)
    labels = gm.fit_predict(X)
    print("\nPCA - EM - 2 Clust")
    print("Homo" + str(homogeneity_score(y, labels)))
    print("Complete" + str((completeness_score(y, labels))))
    print("Rand" + str(adjusted_rand_score(y, labels)))
    print("Silhouette" + str(silhouette_score(X, labels)))
    print("Calinski Harabasz" + str(calinski_harabasz_score(X, labels)))



    ica = FastICA(n_components=37, random_state=42)
    X = ica.fit_transform(R)
    n = 3

    kmeans = KMeans(random_state=42, n_clusters=n)
    kmeans.fit(X)
    print("ICA - KM - " + str(n) + "Clust")
    print("Homo" + str(homogeneity_score(y, kmeans.labels_)))
    print("Complete" + str((completeness_score(y, kmeans.labels_))))
    print("Rand" + str(adjusted_rand_score(y, kmeans.labels_)))
    print("Silhouette" + str(silhouette_score(X, kmeans.labels_)))
    print("Calinski Harabasz" + str(calinski_harabasz_score(X, kmeans.labels_)))

    n=2
    gm = GaussianMixture(n, covariance_type='full', random_state=42)
    labels = gm.fit_predict(X)
    print("\nICA - EM - " + str(n) + " Clust")
    print("Homo" + str(homogeneity_score(y, labels)))
    print("Complete" + str((completeness_score(y, labels))))
    print("Rand" + str(adjusted_rand_score(y, labels)))
    print("Silhouette" + str(silhouette_score(X, labels)))
    print("Calinski Harabasz" + str(calinski_harabasz_score(X, labels)))


    grp = GaussianRandomProjection(n_components=43, random_state=42)
    X = grp.fit_transform(R)
    n = 2

    kmeans = KMeans(random_state=42, n_clusters=n)
    kmeans.fit(X)
    print("RP - KM - " + str(n) + "Clust")
    print("Homo" + str(homogeneity_score(y, kmeans.labels_)))
    print("Complete" + str((completeness_score(y, kmeans.labels_))))
    print("Rand" + str(adjusted_rand_score(y, kmeans.labels_)))
    print("Silhouette" + str(silhouette_score(X, kmeans.labels_)))
    print("Calinski Harabasz" + str(calinski_harabasz_score(X, kmeans.labels_)))

    n = 2
    gm = GaussianMixture(n, covariance_type='full', random_state=42)
    labels = gm.fit_predict(X)
    print("\nRP - EM - " + str(n) + " Clust")
    print("Homo" + str(homogeneity_score(y, labels)))
    print("Complete" + str((completeness_score(y, labels))))
    print("Rand" + str(adjusted_rand_score(y, labels)))
    print("Silhouette" + str(silhouette_score(X, labels)))
    print("Calinski Harabasz" + str(calinski_harabasz_score(X, labels)))




    fa = FeatureAgglomeration(n_clusters=35)
    X = fa.fit_transform(R)
    n = 2

    kmeans = KMeans(random_state=42, n_clusters=n)
    kmeans.fit(X)
    print("FA - KM - " + str(n) + "Clust")
    print("Homo" + str(homogeneity_score(y, kmeans.labels_)))
    print("Complete" + str((completeness_score(y, kmeans.labels_))))
    print("Rand" + str(adjusted_rand_score(y, kmeans.labels_)))
    print("Silhouette" + str(silhouette_score(X, kmeans.labels_)))
    print("Calinski Harabasz" + str(calinski_harabasz_score(X, kmeans.labels_)))

    n = 2
    gm = GaussianMixture(n, covariance_type='full', random_state=42)
    labels = gm.fit_predict(X)
    print("\nFA - EM - " + str(n) + " Clust")
    print("Homo" + str(homogeneity_score(y, labels)))
    print("Complete" + str((completeness_score(y, labels))))
    print("Rand" + str(adjusted_rand_score(y, labels)))
    print("Silhouette" + str(silhouette_score(X, labels)))
    print("Calinski Harabasz" + str(calinski_harabasz_score(X, labels)))






def graphem(X):


    models = [PCA(random_state=42, n_components=34), FastICA(n_components=30, random_state=42),
              GaussianRandomProjection(n_components=45, random_state=42),
              FeatureAgglomeration(n_clusters=35)]
    names = ["PCA", "ICA", "GRP", "FA"]
    c = []
    s = []
    d = []
    N = range(2, 41)

    for x in range(len(models)):

        model = models[x]
        new = model.fit_transform(X)

        silhouette_scores = []
        chs = []
        dbs = []

        for i in N:
            em = GaussianMixture(i, covariance_type='full',random_state=42)
            labels = em.fit_predict(new)
            silhouette_scores.append(silhouette_score(new, labels))
            chs.append(calinski_harabasz_score(new, labels))
            dbs.append(davies_bouldin_score(new, labels))
        s.append(silhouette_scores)
        c.append(chs)
        d.append(dbs)

    plt.close()
    plt.figure()
    for x in range(len(c)):
        plt.plot(N, c[x], label=names[x])
    plt.title('Calinski Harabasz Graph - EM')
    plt.xlabel('Number of clusters')
    plt.ylabel('Calinski Harabasz Score')
    plt.legend()
    plt.savefig('S-3-EM-CHS-Final.png')
    plt.figure()
    plt.close()

    plt.close()
    plt.figure()
    for x in range(len(s)):
        plt.plot(range(2,11), s[x][:9], label=names[x])
    plt.title('Silhouette - EM')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.savefig('S-3-EM-Silhouette-Final.png')
    plt.close()
    plt.figure()





def graphKM(X):

    models = [PCA(random_state=42, n_components=34), FastICA(n_components=30, random_state=42),  GaussianRandomProjection(n_components=45, random_state=42),
              FeatureAgglomeration(n_clusters=35)]
    names = ["PCA", "ICA", "GRP", "FA"]
    w = []
    c = []
    s = []
    N = range(2, 41)

    for x in range(len(models)):

        model = models[x]
        new = model.fit_transform(X)

        wcss = []
        silhouette_scores = []
        chs = []

        for i in N:
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(new)
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(new, kmeans.labels_))
            chs.append(calinski_harabasz_score(new, kmeans.labels_))
        w.append(wcss)
        s.append(silhouette_scores)
        c.append(chs)

    plt.close()
    plt.figure()
    for x in range(len(c)):
        plt.plot(N, c[x], label = names[x])
    plt.title('Calinski Harabasz Graph - K Means')
    plt.xlabel('Number of clusters')
    plt.ylabel('Calinski Harabasz Score')
    plt.legend()
    plt.savefig('S-3-KM-CHS-Final.png')
    plt.figure()
    plt.close()


    plt.close()
    plt.figure()
    for x in range(len(s)):
        plt.plot(range(2,11), s[x][:9], label=names[x])
    plt.title('Silhouette - K Means')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.savefig('S-3-KM-Silhouette-Final.png')
    plt.close()
    plt.figure()



def main():


    datafile = "Data/spambase.data"
    data = np.genfromtxt(datafile, delimiter=",")
    data = data[:-975, :]
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]

    # from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, train_size=.8)

    # from https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data
    sc = MinMaxScaler()
    train_x = sc.fit_transform(train_x)
    test_x = sc.transform(test_x)

    sc = MinMaxScaler()
    X = sc.fit_transform(X)

    graphKM(X)
    graphem(X)

    getScores(X, y)


    pca = PCA(n_components=34, random_state=42)
    projected = pca.fit_transform(X)
    plotkm(projected, y, 3, 'PCA')

    ica = FastICA(n_components=37, random_state=42)
    projected = ica.fit_transform(X)
    plotkm(projected, y, 3, 'ICA')


    grp = GaussianRandomProjection(n_components=43, random_state=42)
    projected = grp.fit_transform(X)
    plotkm(projected, y, 2, 'RP')


    fa = FeatureAgglomeration(n_clusters=35)
    projected = fa.fit_transform(X)
    plotkm(projected, y, 2, 'FA')





    pca = PCA(n_components=34, random_state=42)
    projected = pca.fit_transform(X)
    plotem(projected, y, 3, 'PCA')

    ica = FastICA(n_components=37, random_state=42)
    projected = ica.fit_transform(X)
    plotem(projected, y, 2, 'ICA')

    grp = GaussianRandomProjection(n_components=43, random_state=42)
    projected = grp.fit_transform(X)
    plotem(projected, y, 2, 'RP')

    fa = FeatureAgglomeration(n_clusters=35)
    projected = fa.fit_transform(X)
    plotem(projected, y, 2, 'FA')






if __name__ == "__main__":
    np.random.seed(26)
    random.seed(42)
    main()