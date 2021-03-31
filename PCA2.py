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

from yellowbrick import features
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


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

    errors = []
    for r in range(1,X.shape[1]):
        dr = PCA(random_state=42, n_components=r)
        new = dr.fit_transform(X)
        reconstruct = dr.inverse_transform(new)

        errors.append(mean_squared_error(X, reconstruct))

    plt.plot(range(1, X.shape[1]), errors)
    plt.title('Reconstruction MSE')
    plt.xlabel('Number of Components Kept')
    plt.ylabel('Reconstruction Error')
    plt.savefig('B-PCA-RE.png')
    plt.close()
    plt.figure()


    dr = PCA(random_state=42)
    dr.fit(X)

    plt.close()
    plt.figure()
    fig, ax = plt.subplots()
    fig.suptitle('Eigenvalues and Cumulative Variance')
    ax.plot(dr.explained_variance_ratio_, label="Eigenvalues", color="Blue")
    ax.set_ylabel("Eigenvalue")
    ax.tick_params(axis='y', colors='Blue')
    ax.yaxis.label.set_color("Blue")
    p2 = ax.twinx()
    p2.plot(np.cumsum(dr.explained_variance_ratio_), label="Cumulative Variance", color="red")
    # Color changes from https://stackoverflow.com/questions/1982770/matplotlib-changing-the-color-of-an-axis
    p2.spines['right'].set_color("red")
    p2.spines['left'].set_color("blue")
    p2.tick_params(axis='y', colors='red')
    p2.yaxis.label.set_color("red")
    # legend_handles_labels from https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
    r, y = ax.get_legend_handles_labels()
    r2, y2 = p2.get_legend_handles_labels()
    p2.legend(r + r2, y + y2)
    ax.set_xlabel("Number of Components")
    ax.tick_params(axis='x', labelsize=8)
    p2.set_ylabel("Cumulative Variance")
    # plt.xticks(rotation=45, fontsize=8)
    plt.savefig('B-PCA-Fancy Var Chart.png')


if __name__ == "__main__":
    np.random.seed(26)
    random.seed(42)
    main()

