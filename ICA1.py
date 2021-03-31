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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def main():


    datafile = "Data/spambase.data"
    data = np.genfromtxt(datafile, delimiter=",")
    data = data[:-975, :]
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]

    sc = MinMaxScaler()
    X = sc.fit_transform(X)


    k = []
    for x in range(1,58):
        ica = FastICA(n_components = x, random_state = 42)
        new = ica.fit_transform(X)
        k.append(np.mean(abs(kurtosis(new, axis = 0))))

    plt.plot(range(1, 58), k)
    plt.xlabel('Number of components')
    plt.ylabel('Average Kurtosis')
    plt.title("Average Kurtosis")
    plt.savefig('S-ICA-Kurtosis.png')
    plt.close()
    plt.figure()


    errors = []
    for r in range(1, X.shape[1]):
        dr = FastICA(random_state=42, n_components=r)
        new = dr.fit_transform(X)
        reconstruct = dr.inverse_transform(new)

        errors.append(mean_squared_error(X, reconstruct))

    plt.plot(range(1, X.shape[1]), errors)
    plt.title('Reconstruction MSE')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')
    plt.savefig('S-ICA-RE.png')
    plt.close()
    plt.figure()









if __name__ == "__main__":
    np.random.seed(26)
    random.seed(42)
    main()

