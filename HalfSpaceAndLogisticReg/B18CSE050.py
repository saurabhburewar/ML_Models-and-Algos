import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import linprog


class LogisticReg():
    def __init__(self, learning_rate, iterations):
        self.l_rate = learning_rate
        self.itr = iterations

    def fit(self, X, Y):
        self.samples, self.features = X.shape
        self.W = np.zeros(self.features)
        self.X = X
        self.Y = Y

        self.train_W()

        return self

    def train_W(self):
        for iterr in range(self.itr):
            tot_err = 0
            index = 0
            for sample in self.X:
                p = self.train_predict(sample)
                err = self.Y[index] - p
                tot_err += err**2
                self.W[0] = self.W[0] + self.l_rate * err * p * (1 - p)
                for i in range(self.features-1):
                    self.W[i+1] = self.W[i+1] + self.l_rate * \
                        err * p * (1 - p) * sample[i]
                index += 1

            print("epoch=%d, learning_rate=%.3f, error=%.3f" %
                  (iterr, self.l_rate, tot_err))

        return self

    def train_predict(self, sample):
        p = self.W[0]
        for i in range(self.features-1):
            p += self.W[i + 1] * sample[i]

        return 1 / (1 + np.exp(-p))

    def predict(self, X):
        Y = []
        for sample in X:
            p = self.W[0]
            for i in range(self.features-1):
                p += self.W[i + 1] * sample[i]

            Y.append(round(1 / (1 + np.exp(-p))))

        return Y


# class Half_LP():
#     def __init__(self, learning_rate, iterations):
#         self.l_rate = learning_rate
#         self.itr = iterations

#     def fit(self, X, Y):
#         self.samples, self.features = X.shape
#         self.W = np.zeros(self.features)
#         self.X = X
#         self.Y = Y

#         self.train_W()

#         return self

#     def train_W(self):

#         for sample in X:

#         lp_res = linprog(c=(self.b*self.W), A_ub=lp_x_constraits,
#                          b_ub=lp_y_constraits, options={"disp": True})

#         return self

#     def predict(self, X):
#         Y = []
#         for sample in X:
#             p = self.W[0]
#             for i in range(self.features-1):
#                 p += self.W[i + 1] * sample[i]

#             Y.append(p)

#         return Y


class Half_percep():
    def __init__(self, learning_rate, iterations):
        self.l_rate = learning_rate
        self.itr = iterations

    def fit(self, X, Y):
        self.samples, self.features = X.shape
        self.W = np.zeros(self.features)
        self.X = X
        self.Y = Y

        self.train_W()

        return self

    def train_W(self):
        for iterr in range(self.itr):
            tot_err = 0
            index = 0
            for sample in self.X:
                p = self.train_predict(sample)
                err = sample[-1] - p
                tot_err += err**2
                self.W[0] = self.W[0] + self.l_rate * err
                for i in range(self.features-1):
                    self.W[i+1] = self.W[i+1] + \
                        (2048 * self.l_rate * err * sample[i])
                index += 1

            print("epoch=%d, learning_rate=%.3f, error=%.3f" %
                  (iterr, self.l_rate, tot_err))

        return self

    def train_predict(self, sample):
        activ = self.W[0]
        for i in range(self.features-1):
            activ += self.W[i + 1] * float(sample[i])
        if activ >= 0:
            return 1
        else:
            return 0

    def predict(self, X):
        Y = []
        for sample in X:
            activ = self.W[0]
            for i in range(self.features-1):
                activ += self.W[i + 1] * sample[i]
            if activ >= 0:
                Y.append(1)
            else:
                Y.append(0)

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

    # Count the missing values
    # print(df.isnull().sum().sort_values(ascending=False))

    # There are a lot of missing values in the column 'Cabin' and we are not
    # going to use it, so we will drop it before dropping all NaNs.
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
    print("        Half space classifier using LP solver        ")
    print("-----------------------------------------------------")

    print("=====================================================")
    print("       Half space classifier using Perceptron        ")
    print("-----------------------------------------------------")

    model1 = Half_percep(learning_rate=0.01, iterations=100)

    model1.fit(X_train, Y_train)
    Y_pred = model1.predict(X_test)
    model_accuracy(Y_test, Y_pred)

    # print("=====================================================")
    # print("           Logistic Regression Classifier            ")
    # print("-----------------------------------------------------")

    # model2 = LogisticReg(learning_rate=0.01, iterations=10)

    # model2.fit(X_train, Y_train)
    # Y_pred = model2.predict(X_test)
    # model_accuracy(Y_test, Y_pred)

    print("=====================================================")


if __name__ == "__main__":
    main()
