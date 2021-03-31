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

    X = train_x


    for y in range(10):

        errors = []
        for x in range (1,58):

            rca = GaussianRandomProjection(n_components=x, random_state=y)
            X_train_pca = rca.fit_transform(X)
            reconstructed = np.dot(X_train_pca, np.linalg.pinv(rca.components_).T)

            errors.append(mean_squared_error(X, reconstructed))


        plt.plot(range (1,58), errors)

    plt.xlabel('Number of components')
    plt.ylabel('Reconstruction MSE')
    plt.title("Reconstruction MSE")
    plt.savefig('S-RCA-Reconstruction.png')
    plt.close()
    plt.figure()


if __name__ == "__main__":
    np.random.seed(26)
    random.seed(42)
    main()