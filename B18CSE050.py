import numpy as np
from numpy.linalg import inv
from numpy.linalg import qr
import pandas as pd
from sklearn.model_selection import train_test_split


class LinearReg_PI():
    def __init__(self, learning_rate, iterations):
        self.l_rate = learning_rate
        self.itr = iterations

    def fit(self, X, Y):
        self.samples, self.features = X.shape
        self.W = np.zeros(self.features)
        self.X = X
        self.Y = Y

        self.train()

        return self

    def train(self):
        for iterr in range(self.itr):
            Q, R = qr(self.X)
            self.W = inv(R).dot(Q.T).dot(self.Y)
            p = self.X.dot(self.W)

            loss = np.mean((p - self.Y)**2)

            print("epoch=%d, learning_rate=%.3f, loss=%.3f" %
                  (iterr, self.l_rate, loss))

        return self

    def predict(self, X):
        Y = X.dot(self.W)
        Y = Y.astype(int)
        Y[Y < 0] = 0
        return Y


class LinearReg_GD():
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
        for iterr in range(self.itr):
            p = np.dot(self.X, self.W) + self.b
            loss = np.mean((p - self.Y)**2)

            dw = (1/self.samples)*np.dot(self.X.T, (p - self.Y))
            db = (1/self.samples)*np.sum((p - self.Y))

            self.W -= self.l_rate*dw
            self.b -= self.l_rate*db

            print("epoch=%d, learning_rate=%.3f, loss=%.3f" %
                  (iterr, self.l_rate, loss))

        return self

    def predict(self, X):
        Y = np.dot(X, self.W) + self.b
        Y = Y.astype(int)
        Y[Y < 0] = 0
        return Y


def model_accuracy(Y_test, Y_pred):
    correctly_classified = 0
    for count in range(np.size(Y_pred)):
        if Y_test[count] == Y_pred[count]:
            correctly_classified = correctly_classified + 1
        count = count + 1

    print("Accuracy on test set by this model       :  %.3f" %
          ((correctly_classified / count) * 100))


def main():
    df = pd.read_csv('./Data/train.csv')

    df.drop(['Cabin'], axis=1, inplace=True)

    df.dropna(how='any', inplace=True)
    df['sex_factor'] = pd.factorize(df.Sex)[0]
    df['em_factor'] = pd.factorize(df.Embarked)[0]

    X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'sex_factor', 'em_factor']]
    Y = df['Survived']
    X = X.values
    Y = Y.values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    print("=====================================================")
    print("    Linear Regression Classifier - Pseudo Inverse    ")
    print("-----------------------------------------------------")

    model = LinearReg_PI(learning_rate=0.01, iterations=10)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    model_accuracy(Y_test, Y_pred)

    print("=====================================================")
    print("   Linear Regression Classifier - Gradient Descent   ")
    print("-----------------------------------------------------")

    model1 = LinearReg_GD(learning_rate=0.01, iterations=10)
    model1.fit(X_train, Y_train)
    Y_pred = model1.predict(X_test)
    model_accuracy(Y_test, Y_pred)

    print("=====================================================")


if __name__ == "__main__":
    main()
