import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


class Kmeans():
    def __init__(self, k, itr=300):
        self.k = k
        self.itr = itr

    def fit(self, data):
        self.centroids = {}
        self.labels = [None] * data.shape[0]

        # select first two examples as centroids to get random centroids
        # np.random.shuffle(data)
        for i in range(self.k):
            self.centroids[i] = data[i]

        # Start iteration loop
        for i in range(self.itr):
            self.clusters = {}

            for i in range(self.k):
                self.clusters[i] = []

            # Put all examples into clusters
            for index, row in enumerate(data):
                distances = [np.linalg.norm(row-self.centroids[c])
                             for c in self.centroids]
                cluster = distances.index(min(distances))
                self.clusters[cluster].append(row)
                self.labels[index] = cluster

            # Find new centroids by finding mean of clusters
            for cluster in self.clusters:
                self.centroids[cluster] = np.average(
                    self.clusters[cluster], axis=0)

    def predict(self, data):
        distances = []
        for c in centroids:
            distances.append((sum((c - data)**2))**0.5)
            cluster = distances.index(min(distances))

        return cluster


class GMM():
    def __init__(self, k, itr=100, tol=1e-4):
        self.k = k
        self.itr = itr
        self.tol = tol

    def fit(self, X, start=None):
        self.r = np.zeros((X.shape[0], self.k))

        np.random.seed(4)

        if start == None:
            random_row = np.random.randint(low=0, high=X.shape[0], size=self.k)
            self.meanV = [X[row_index, :] for row_index in random_row]
        else:
            random_row = np.random.randint(low=0, high=X.shape[0], size=self.k)
            self.meanV = [start[c] for c in start]

        self.W = np.full(self.k, 1/self.k)
        shape = self.k, X.shape[1], X.shape[1]
        self.covM = np.full(shape, np.cov(X, rowvar=False))
        log_l = 0
        self.converged = False
        self.log_trace = []

        for i in range(self.itr):

            # Estimation
            for j in range(self.k):
                prior = self.W[j]
                likeli = multivariate_normal(
                    self.meanV[j], self.covM[j]).pdf(X)
                self.r[:, j] = prior * likeli

            log_l_new = np.sum(np.log(np.sum(self.r, axis=1)))
            self.r = self.r / self.r.sum(axis=1, keepdims=1)

            # Maximisation
            r_w = self.r.sum(axis=0)

            self.W = r_w / X.shape[0]

            w_sum = np.dot(self.r.T, X)
            self.meanV = w_sum / r_w.reshape(-1, 1)

            for j in range(self.k):
                diff = (X - self.meanV[j]).T
                w_sum = np.dot(self.r[:, j] * diff, diff.T)
                self.covM[j] = w_sum / r_w[j]

            # Checking convergence
            if abs(log_l_new - log_l) <= self.tol:
                self.converged = True
                break

            log_l = log_l_new
            self.log_trace.append(log_l)

        return self


def runkmeans(X):
    colors = 10*["g", "r", "c", "b", "k"]

    sslist = {}

    for x in range(2, 10):
        clf = Kmeans(k=x)
        clf.fit(X)
        ss = silhouette_score(X, clf.labels)
        sslist[x] = ss

    if not os.path.isdir('B18CSE050_Plots'):
        os.mkdir('B18CSE050_Plots')

    plt.figure(2)
    plt.plot(sslist.keys(), sslist.values())
    plt.title('Silhouette score for various k values')
    plt.xlabel('K')
    plt.ylabel('Silhouette score')
    plt.savefig('B18CSE050_plots/Silhoutte.png')

    op_k = max(sslist, key=sslist.get)
    clf = Kmeans(k=op_k)
    clf.fit(X)

    plt.figure(3)
    for cluster in clf.clusters:
        color = colors[cluster]
        for featureset in clf.clusters[cluster]:
            plt.scatter(featureset[0], featureset[1],
                        color=color, s=10, linewidths=5)

    for centroid in clf.centroids:
        plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                    marker="o", color="k", s=100, linewidths=5)

    plt.title('Clusters for the best k value')
    plt.savefig('B18CSE050_plots/Kmeans.png')


def runGMM(X):
    sslist = {}

    for x in range(2, 10):
        clf = Kmeans(k=x)
        clf.fit(X)
        ss = silhouette_score(X, clf.labels)
        sslist[x] = ss

    op_k = max(sslist, key=sslist.get)
    clf = Kmeans(k=op_k)
    clf.fit(X)

    gmm = GMM(k=2)
    gmm.fit(X, clf.centroids)

    plt.figure(4)
    plt.plot(X[:, 0], X[:, 1], 'ko')

    delta = 0.025
    k = gmm.meanV.shape[0]
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = gmm.meanV[i]
        cov = gmm.covM[i]
        z_grid = multivariate_normal(mean, cov).pdf(
            coordinates).reshape(x_grid.shape)
        plt.contour(x_grid, y_grid, z_grid, colors=col[i])

    plt.title('GMM Clusters')
    plt.tight_layout()
    plt.savefig('B18CSE050_plots/GMM_after_Kmeans.png')


def runGMMwithCov(data):

    cov_types = ['full', 'diag', 'spherical']
    k = 3

    for i in range(3):
        gmm = GaussianMixture(n_components=k, covariance_type=cov_types[i])
        gmm.fit(data)

        plt.figure(5)
        labels = gmm.predict(data)
        frame = pd.DataFrame(data)
        frame['cluster'] = labels
        frame.columns = ['Weight', 'Height', 'cluster']

        color = ['red', 'green', 'blue', 'cyan', 'pink', 'black']
        for j in range(0, k):
            data = frame[frame["cluster"] == j]
            plt.scatter(data["Weight"], data["Height"], c=color[j])

        plt.savefig('B18CSE050_plots/GMM_{}.png'.format(cov_types[i]))


def PCA(X, Y):
    X_s = StandardScaler().fit_transform(X)

    CovM = np.cov(X_s.T)
    e_values, e_vectors = np.linalg.eig(CovM)

    var = []
    for i in range(len(e_values)):
        var.append(e_values[i]/np.sum(e_values))

    X_PCA = X_s.dot(e_vectors.T[0])
    Y_PCA = X_s.dot(e_vectors.T[1])

    res = pd.DataFrame(X_PCA, columns=['X_PCA'])
    res['Y_PCA'] = Y_PCA
    res['output'] = Y

    plt.figure(1)
    plt.title('PCA plot of the data')
    # plt.scatter(res['plot_x'], res['plot_y'], c=res['output'])
    sns.scatterplot(x=res['X_PCA'], y=res['Y_PCA'], hue=res['output'], s=100)
    plt.savefig('B18CSE050_plots/PCA_plot.png')

    return res


def getData1():
    df = pd.read_csv(
        'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')

    X = df.drop(['species'], axis=1)
    Y = df.loc[:, 'species']

    return X, Y


def getData2():
    df = pd.read_csv('./titanic.csv')

    df.drop(['Cabin'], axis=1, inplace=True)
    df.dropna(how='any', inplace=True)
    df['sex_factor'] = pd.factorize(df.Sex)[0]
    df['em_factor'] = pd.factorize(df.Embarked)[0]

    X = df[['Pclass', 'Age', 'Fare', 'sex_factor', 'em_factor']]
    Y = df['Survived']

    return X, Y


def getData3():
    df = pd.read_csv('./data.csv')

    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': -1})
    df.drop(['Unnamed: 32'], axis=1, inplace=True)
    df.drop(['id'], axis=1, inplace=True)

    X = df.drop(['diagnosis'], axis=1)
    Y = df.loc[:, 'diagnosis']

    return X, Y


def main():

    X, Y = getData1()
    # X, Y = getData2()
    # X, Y = getData3()

    res = PCA(X, Y)
    X_PCA = res.drop(['output'], axis=1)
    Y_PCA = res.loc[:, 'output']

    X_PCA = X_PCA.values
    Y_PCA = Y_PCA.values

    print("=====================================================")
    print("           K-means with silhouette score             ")
    print("-----------------------------------------------------")

    runkmeans(X_PCA)
    print('Done \nPlease check "B18CSE050_plots" directory for the plots')

    print("=====================================================")
    print("   GMM using K-means result as starting centroids    ")
    print("-----------------------------------------------------")

    runGMM(X_PCA)
    print('Done \nPlease check "B18CSE050_plots" directory for the plots')

    print("=====================================================")
    print("        GMM with different covariance matrix         ")
    print("-----------------------------------------------------")

    runGMMwithCov(X_PCA)
    print('Done \nPlease check "B18CSE050_plots" directory for the plots')

    print("=====================================================")


if __name__ == "__main__":
    main()
