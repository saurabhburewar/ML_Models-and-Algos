import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm, metrics
import cvxopt as opt


class Hard_SVM():
    def __init__(self, C=None):
        self.C = C

    def fit(self, X, Y):
        self.samples, self.features = X.shape
        self.W = np.zeros(self.features)
        self.b = float(0)
        self.X = X
        self.Y = Y

        self.train()

        return self

    def train(self):
        y = self.Y.reshape(-1, 1) * 1
        X_dash = y * self.X

        K = np.zeros((self.samples, self.samples))
        for i in range(self.samples):
            for j in range(self.samples):
                K[i, j] = np.dot(self.X[i], self.X[j])

        H = np.dot(X_dash, X_dash.T) * 1

        P = opt.matrix(H)
        q = opt.matrix(-np.ones((self.samples, 1)))
        G = opt.matrix(-np.eye(self.samples))
        h = opt.matrix(np.zeros(self.samples))
        A = opt.matrix(y.reshape(1, -1).astype(float))
        b = opt.matrix(np.zeros(1))

        opt.solvers.options['show_progress'] = False

        sol = opt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(sol['x'])

        S = alphas > 1e-4
        self.a = alphas[S]
        self.sv = self.X[S]
        self.sv_y = y[S]
        ind = np.arange(len(alphas))[S]

        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], S])
        self.b /= len(self.a)

        for n in range(len(self.a)):
            self.W += self.a[n] * self.sv_y[n] * self.sv[n]

    def predict(self, x):
        return np.sign(np.dot(x, self.W) + self.b)


class Soft_SVM():
    def __init__(self, C=10):
        self.C = float(C)

    def fit(self, X, Y):
        self.samples, self.features = X.shape
        self.W = np.zeros(self.features)
        self.b = float(0)
        self.X = X
        self.Y = Y

        self.train()

        return self

    def train(self):
        y = self.Y.reshape(-1, 1) * 1
        X_dash = y * self.X

        K = np.zeros((self.samples, self.samples))
        for i in range(self.samples):
            for j in range(self.samples):
                K[i, j] = np.dot(self.X[i], self.X[j])

        H = np.dot(X_dash, X_dash.T) * 1

        P = opt.matrix(H)
        q = opt.matrix(-np.ones((self.samples, 1)))
        G = opt.matrix(
            np.vstack((np.eye(self.samples)*-1, np.eye(self.samples))))
        h = opt.matrix(
            np.hstack((np.zeros(self.samples), np.ones(self.samples) * self.C)))
        A = opt.matrix(y.reshape(1, -1).astype(float))
        b = opt.matrix(np.zeros(1))

        opt.solvers.options['show_progress'] = False

        sol = opt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(sol['x'])

        S = alphas > 1e-4
        self.a = alphas[S]
        self.sv = self.X[S]
        self.sv_y = y[S]
        ind = np.arange(len(alphas))[S]

        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], S])
        self.b /= len(self.a)

        for n in range(len(self.a)):
            self.W += self.a[n] * self.sv_y[n] * self.sv[n]

    def predict(self, x):
        return np.sign(np.dot(x, self.W) + self.b)


class Soft_SVM_SGD():
    def __init__(self, learning_rate, iterations):
        self.l_rate = learning_rate
        self.itr = iterations

    def fit(self, X, Y):
        self.samples, self.features = X.shape
        self.W = np.zeros(self.features)
        self.b = 0
        self.X = X
        self.Y = Y

        self.train()

        return self

    def train(self):
        eta = 1

        for itrr in range(1, self.itr):
            for i, x in enumerate(self.X):
                if (self.Y[i]*np.dot(self.X[i], self.W)) < 1:
                    self.W = self.W + eta * \
                        ((self.X[i] * self.Y[i]) + (-2 * (1/itrr) * self.W))
                else:
                    self.W = self.W + eta * (-2 * (1/itrr) * self.W)

        return self

    def predict(self, x):
        return np.sign(np.dot(x, self.W) + self.b)


def main():
    df = pd.read_csv('./Data/data.csv')

    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': -1})
    df.drop(['Unnamed: 32'], axis=1, inplace=True)
    df.drop(['id'], axis=1, inplace=True)

    X = df.drop(['diagnosis'], axis=1)
    Y = df.loc[:, 'diagnosis']
    X_nor = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_nor)
    X.insert(loc=len(X.columns), column='intercept', value=1)
    X = X.astype(float)

    X = X.values
    Y = Y.values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    print("=====================================================")

    print("Data is not linearly seperable")

    print("=====================================================")
    print("                   Hard SVM by QP                    ")
    print("-----------------------------------------------------")

    model1 = Hard_SVM()
    model1.fit(X_train, Y_train)
    Y_pred1 = model1.predict(X_test)
    print("Accuracy of the model ",
          metrics.accuracy_score(Y_test, Y_pred1))

    print("=====================================================")
    print("                   Soft SVM by QP                    ")
    print("-----------------------------------------------------")

    model2 = Soft_SVM()
    model2.fit(X_train, Y_train)
    Y_pred2 = model2.predict(X_test)
    print("Accuracy of the model ",
          metrics.accuracy_score(Y_test, Y_pred2))

    print("=====================================================")
    print("                  Soft SVM by SGD                    ")
    print("-----------------------------------------------------")

    model3 = Soft_SVM_SGD(learning_rate=0.01, iterations=10)
    model3.fit(X_train, Y_train)
    Y_pred3 = model3.predict(X_test)
    print("Accuracy of the model ",
          metrics.accuracy_score(Y_test, Y_pred3))

    print("=====================================================")


if __name__ == "__main__":
    main()
