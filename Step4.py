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
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import OneHotEncoder


# From https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/
def plotLearningCurves(train_sizes, train_scores, test_scores, title, name):

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="b", label="Training score")
    plt.plot(train_sizes, test_mean, color="g", label="Cross-validation score")

    # # Draw bands
    # plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    # plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title(title)
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.yticks(np.arange(.7, 1, .02))
    plt.tight_layout()
    plt.savefig(name)
    plt.close()
    plt.figure()

#From https://www.geeksforgeeks.org/validation-curve/
def plotValidationCurves(train_score, test_score, title, name, param_name, parameter_range):

    # Calculating mean and standard deviation of training score
    mean_train_score = np.mean(train_score, axis=1)
    std_train_score = np.std(train_score, axis=1)

    # Calculating mean and standard deviation of testing score
    mean_test_score = np.mean(test_score, axis=1)
    std_test_score = np.std(test_score, axis=1)

    # Plot mean accuracy scores for training and testing scores
    plt.plot(parameter_range, mean_train_score,
             label="Training Score", color='b')
    plt.plot(parameter_range, mean_test_score,
             label="Cross Validation Score", color='g')

    # Creating the plot
    plt.title(title)
    plt.xlabel(param_name)
    # plt.yticks(np.arange(.89, 1, .02))
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig(name)
    plt.close()
    plt.figure()


def tuning(train_x, train_y, name):


    train_sizes, train_scores, test_scores = learning_curve(MLPClassifier(random_state=42), train_x, train_y,cv=5,
                                                            scoring='accuracy',
                                                            n_jobs=-1)

    plotLearningCurves(train_sizes, train_scores, test_scores, title = "NN Learning Curve Default Parameters", name = "ReportNNSpam LC 1" )

    # parameter_range =[.0001, .001,  .005, .01, 1, 5]#range(1,11)
    # train_score, test_score = validation_curve(MLPClassifier(random_state = 42), train_x, train_y,
    #                                            param_name="alpha",
    #                                            param_range=parameter_range,
    #                                            cv=5, scoring="accuracy")
    #
    # plotValidationCurves(train_score, test_score, title="NN alpha Validation Curve", name="B-4-"+name+" VC 1",
    #                      param_name="alpha", parameter_range = parameter_range)
    #

    # # parameter_range = [(50,100,), (20, 50), (10,30), (30,70),(100,100), (10,20), (20,20), (150,)]
    # # parameter_range = [(10, 10,), (10, 20), (10, 30), (10, 50), (10, 100), (10, 150)]
    # parameter_range = [(1, ),(10,),(20,), (30,), (50,), (100,), (150)]
    # # parameter_range = ['(10,10)', '(10,20)', '(10,30)', '(10,40)', '(10,50)','(5,10)', '(5,20)', '(50,)', '(100,)']
    # train_score, test_score = validation_curve(MLPClassifier(random_state=42, alpha=.001), train_x, train_y,
    #                                            param_name="hidden_layer_sizes",
    #                                            param_range=parameter_range,
    #                                            cv=5, scoring="accuracy")
    # # parameter_range = ['(10, 10,)', '(10, 20)', '(10, 30)', '(10, 50)', '(10, 100)', '(10, 150)']
    # parameter_range = ['(1,)', '(10,)', '(20,)', '(30,)', '(50,)', '(100,)', '(150)']
    # plotValidationCurves(train_score, test_score, title="NN Validation Curve Layers alpha = .001", name="B-4-"+name+" VC 3",
    #                      param_name="# nodes", parameter_range=parameter_range)
    #

    # parameter_range = range(1, 200, 10)
    # lrs = [.0005, .001, .005]
    # colors = ['green', 'blue', 'red']
    # for x in range(len(lrs)):
    #     train_score, test_score = validation_curve(
    #         MLPClassifier(random_state=42, hidden_layer_sizes=(10, 150), alpha=.001, learning_rate_init=lrs[x]), train_x,
    #         train_y,
    #         param_name="max_iter",
    #         param_range=parameter_range,
    #         cv=5, scoring="accuracy")
    #
    #     plt.plot(parameter_range, np.mean(train_score, axis=1),
    #              label='Train LR ={0}'.format(lrs[x]), color=colors[x])
    #     plt.plot(parameter_range, np.mean(test_score, axis=1), '--',
    #              label='CV LR={0}'.format(lrs[x]), color=colors[x])
    #
    # plt.title("Learning Rate Learning Curves")
    # plt.xlabel("Iteration")
    # plt.ylabel("Accuracy")
    # plt.ylim(.87, .95)
    # plt.tight_layout()
    # plt.legend(loc='best', prop={'size': 10})
    # plt.savefig('S-4-'+name+' VC 5')
    # plt.close()
    # plt.figure()



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
    sc = StandardScaler()
    otrain = sc.fit_transform(train_x)
    otest = sc.transform(test_x)
    sc = MinMaxScaler()
    train_x = sc.fit_transform(otrain)
    test_x = sc.transform(otest)



    pca = PCA(n_components=20, random_state=42)
    projectedPCA = pca.fit_transform(train_x)
    testPCA = pca.transform(test_x)

    ica = FastICA(n_components=25, random_state=42)
    projectedICA = ica.fit_transform(train_x)
    testICA = ica.transform(test_x)


    grp = GaussianRandomProjection(n_components=35, random_state=42)
    projectedGRP = grp.fit_transform(train_x)
    testGRP = grp.transform(test_x)

    fa = FeatureAgglomeration(n_clusters=22)
    projectedFA = fa.fit_transform(train_x)
    testFA = fa.transform(test_x)

    data = [otrain, projectedPCA, projectedICA, projectedGRP, projectedFA]
    test_data = [otest, testPCA, testICA, testGRP, testFA]
    names = ["Original", "PCA", "ICA", "GRP", "FA"]

    # KMmodels = [MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
    #                          max_iter=50),
    #           MLPClassifier(alpha=.001, hidden_layer_sizes=(20,), random_state=42, learning_rate_init=.005,
    #                         max_iter=30),
    #           MLPClassifier(alpha=.001, hidden_layer_sizes=(150,), random_state=42, learning_rate_init=.005,
    #                         max_iter=40),
    #           MLPClassifier(alpha=.001, hidden_layer_sizes=(50,), random_state=42, learning_rate_init=.005,
    #                         max_iter=30),
    #           MLPClassifier(alpha=.001, hidden_layer_sizes=(50,), random_state=42, learning_rate_init=.005,
    #                         max_iter=30),
    #           ]
    # EMmodels = [MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
    #                         max_iter=50),
    #           MLPClassifier(alpha=.001, hidden_layer_sizes=(20,), random_state=42, learning_rate_init=.005,
    #                         max_iter=30),
    #           MLPClassifier(alpha=.001, hidden_layer_sizes=(150,), random_state=42, learning_rate_init=.005,
    #                         max_iter=40),
    #           MLPClassifier(alpha=.001, hidden_layer_sizes=(50,), random_state=42, learning_rate_init=.005,
    #                         max_iter=30),
    #           MLPClassifier(alpha=.001, hidden_layer_sizes=(50,), random_state=42, learning_rate_init=.005,
    #                         max_iter=30),
    #           ]
    # Regmodels = [MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
    #                           max_iter=50),
    #             MLPClassifier(alpha=.001, hidden_layer_sizes=(20,), random_state=42, learning_rate_init=.005,
    #                           max_iter=30),
    #             MLPClassifier(alpha=.001, hidden_layer_sizes=(150,), random_state=42, learning_rate_init=.005,
    #                           max_iter=40),
    #             MLPClassifier(alpha=.001, hidden_layer_sizes=(50,), random_state=42, learning_rate_init=.005,
    #                           max_iter=30),
    #             MLPClassifier(alpha=.001, hidden_layer_sizes=(50,), random_state=42, learning_rate_init=.005,
    #                           max_iter=30),
    #             ]

    KMmodels = [MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                              max_iter=50),
                MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                              max_iter=50),
                MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                              max_iter=50),
                MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                              max_iter=50),
                MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                              max_iter=50),
                ]
    EMmodels = [MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                              max_iter=50),
                MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                              max_iter=50),
                MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                              max_iter=50),
                MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                              max_iter=50),
                MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                              max_iter=50),
                ]
    Regmodels = [MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                              max_iter=50),
                 MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                               max_iter=50),
                 MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                               max_iter=50),
                 MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                               max_iter=50),
                 MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                               max_iter=50),
                 ]

    JustEMmodels = [MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                              max_iter=50),
                MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                              max_iter=50),
                MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                              max_iter=50),
                MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                              max_iter=50),
                MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                              max_iter=50),
                ]
    JustKMmodels = [MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                               max_iter=50),
                 MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                               max_iter=50),
                 MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                               max_iter=50),
                 MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                               max_iter=50),
                 MLPClassifier(alpha=.001, hidden_layer_sizes=(200,), random_state=42, learning_rate_init=.005,
                               max_iter=50),
                 ]

    KMclust = [KMeans(n_clusters=2, random_state=42), KMeans(n_clusters=2, random_state=42),
             KMeans(n_clusters=5, random_state=42), KMeans(n_clusters=2, random_state=42),
             KMeans(n_clusters=2, random_state=42)]
    EMclust = [GaussianMixture(n_components=2, random_state=42), GaussianMixture(n_components=2, random_state=42),
               GaussianMixture(n_components=2, random_state=42),GaussianMixture(n_components=2, random_state=42),
               GaussianMixture(n_components=2, random_state=42)]

    KM = [KMeans(n_clusters=2, random_state=42), KMeans(n_clusters=2, random_state=42),
               KMeans(n_clusters=5, random_state=42), KMeans(n_clusters=2, random_state=42),
               KMeans(n_clusters=2, random_state=42)]
    EM = [GaussianMixture(n_components=2, random_state=42), GaussianMixture(n_components=2, random_state=42),
               GaussianMixture(n_components=2, random_state=42), GaussianMixture(n_components=2, random_state=42),
               GaussianMixture(n_components=2, random_state=42)]

    KMtrain_times = []

    EMtrain_times = []

    JustKMtrain_times = []

    JustEMtrain_times = []

    Regtrain_times = []




    KMtest_scores = []

    EMtest_scores = []

    JustKMtest_scores = []

    JustEMtest_scores = []

    Regtest_scores=[]



    for x in range(len(data)):

        for r in range(3):

            training_data = data[x]
            testing_data = test_data[x]
            model = Regmodels[x]
            train_times = Regtrain_times
            test_scores = Regtest_scores

            if r == 0:
                cluster = KMclust[x]

                train_labels = cluster.fit_predict(data[x])
                # train_labels = np.reshape(train_labels, (data[x].shape[0], 1))
                # # https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
                # enc = OneHotEncoder(handle_unknown='ignore')
                # labels = enc.fit_transform(train_labels).toarray()
                # new_train = np.append(data[x], labels, 1)
                # training_data = new_train


                labels = np.reshape(train_labels, (data[x].shape[0], 1))
                new_train = np.append(data[x], labels, 1)
                training_data = new_train

                test_labels = cluster.predict(test_data[x])
                test_labels = np.reshape(test_labels, (test_data[x].shape[0], 1))
                # # https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
                # enc = OneHotEncoder(handle_unknown='ignore')
                # labels = enc.fit_transform(test_labels).toarray()
                # new_test = np.append(test_data[x], labels, 1)
                # testing_data = new_test


                labels = np.reshape(test_labels, (test_data[x].shape[0], 1))
                new_test = np.append(test_data[x], labels, 1)
                testing_data = new_test

                model = KMmodels[x]
                train_times = KMtrain_times
                test_scores = KMtest_scores


            elif r == 1:
                cluster = EMclust[x]
                train_labels = cluster.fit_predict(data[x])
                # train_labels = np.reshape(train_labels, (data[x].shape[0], 1))
                # #https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
                # enc = OneHotEncoder(handle_unknown='ignore')
                # labels = enc.fit_transform(train_labels).toarray()
                # new_train = np.append(data[x], labels, 1)
                # training_data = new_train

                labels = np.reshape(train_labels, (data[x].shape[0], 1))
                new_train = np.append(data[x], labels, 1)
                training_data = new_train


                test_labels = cluster.predict(test_data[x])
                # test_labels = np.reshape(test_labels, (test_data[x].shape[0], 1))
                # # https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
                # enc = OneHotEncoder(handle_unknown='ignore')
                # labels = enc.fit_transform(test_labels).toarray()
                # new_test = np.append(test_data[x], labels, 1)
                # if names[x] == "ICA":
                #     b = np.zeros((new_test.shape[0], new_test.shape[1] + 1))
                #     b[:, :-1] = new_test
                #     new_test = b
                #
                # testing_data = new_test

                labels = np.reshape(test_labels, (test_data[x].shape[0], 1))
                new_test = np.append(test_data[x], labels, 1)
                testing_data = new_test

                model = EMmodels[x]
                train_times = EMtrain_times
                test_scores = EMtest_scores

            if r == 3:
                cluster = KM[x]

                train_labels = cluster.fit_predict(data[x])
                train_labels = np.reshape(train_labels, (data[x].shape[0], 1))
                # https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
                enc = OneHotEncoder(handle_unknown='ignore')
                labels = enc.fit_transform(train_labels).toarray()
                training_data = labels

                test_labels = cluster.predict(test_data[x])
                test_labels = np.reshape(test_labels, (test_data[x].shape[0], 1))
                # https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
                enc = OneHotEncoder(handle_unknown='ignore')
                labels = enc.fit_transform(test_labels).toarray()
                testing_data = labels

                model = JustKMmodels[x]
                train_times = JustKMtrain_times
                test_scores = JustKMtest_scores

            if r == 4:
                cluster = EM[x]
                train_labels = cluster.fit_predict(data[x])
                train_labels = np.reshape(train_labels, (data[x].shape[0], 1))
                #https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
                enc = OneHotEncoder(handle_unknown='ignore')
                labels = enc.fit_transform(train_labels).toarray()
                training_data = labels

                test_labels = cluster.predict(test_data[x])
                test_labels = np.reshape(test_labels, (test_data[x].shape[0], 1))
                # https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
                enc = OneHotEncoder(handle_unknown='ignore')
                labels = enc.fit_transform(test_labels).toarray()
                if names[x] == "ICA":
                    b = np.zeros((labels.shape[0], labels.shape[1] + 1))
                    b[:, :-1] = labels
                    labels = b

                testing_data = labels


                model = JustEMmodels[x]
                train_times = JustEMtrain_times
                test_scores = JustEMtest_scores


            print(names[x])
            start = time.time()
            model.fit(training_data, train_y)
            end = time.time()

            train_times.append((end - start))
            print(testing_data.shape)
            y_pred = model.predict(testing_data)
            test_score = accuracy_score(test_y, y_pred)
            test_scores.append(test_score)


    #Plotting from https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    vals = [KMtrain_times, EMtrain_times, Regtrain_times]#, JustKMtrain_times, JustEMtrain_times]
    X = names
    width = 0.8
    color_list = ['b', 'g', 'r']#, 'y', 'c']
    words = ["KM", "EM", "None"]#,"Just KM Clusters", "Just EM Clusters"]

    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(_X - width / 2. + i / float(n) * width, vals[i],
                width=width / float(n), align="edge",color=color_list[i % len(color_list)],
                label = words[i])
    plt.xticks(_X, X)
    plt.ylabel("Train Time")
    plt.legend()
    plt.title("Clustering vs Train Time")
    # plt.ylim(.15, .3)
    plt.savefig("PT4 Train Times.png")
    plt.close()
    plt.figure()

    # Plotting from https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    vals = [KMtest_scores, EMtest_scores, Regtest_scores]#, JustKMtest_scores, JustEMtest_scores]
    X = names
    width = 0.8
    color_list = ['b', 'g', 'r']#, 'y', 'c']
    words = ["Original + KM", "Original + EM", "Original"]#, "Just KM Clusters", "Just EM Clusters"]

    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(_X - width / 2. + i / float(n) * width, vals[i],
                width=width / float(n), align="edge", color=color_list[i % len(color_list)],
                label=words[i])
    plt.xticks(_X, X)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.ylim(.8, .95)
    plt.title("Clustering vs Accuracy")
    plt.savefig("PT4 Test Scores.png")




if __name__ == "__main__":
    np.random.seed(26)
    random.seed(42)
    main()