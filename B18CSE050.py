import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class Adaboost():

    def __init__(self):
        self.stumps = []

    def fit(self, X, Y, stages=100):
        self.alphas = []
        self.errors = []
        self.stages = stages
        W = np.array([1/len(Y) for i in range(len(Y))])

        # plotx_min, plotx_max = X[:, 0].min(), X[:, 0].max()
        # ploty_min, ploty_max = X[:, 1].min(), X[:, 1].max()
        # xx, yy = np.meshgrid(np.arange(plotx_min, plotx_max, 0.02),
        #                      np.arange(ploty_min, ploty_max, 0.02))

        for stage in range(0, self.stages):

            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, Y, sample_weight=W)
            Y_pred = stump.predict(X)

            # Z = stump.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            # Z = Z.reshape(xx.shape)
            # plt.contour(xx, yy, Z, alpha=.2)

            self.stumps.append(stump)

            stump_e = (
                sum(W * (np.not_equal(Y, Y_pred)).astype(int)))/sum(W)
            self.errors.append(stump_e)

            stump_alpha = np.log((1 - stump_e) / stump_e)
            self.alphas.append(stump_alpha)

            W = W * np.exp(stump_alpha * (np.not_equal(Y, Y_pred)).astype(int))

            print("Stage = {stage}, alpha = {alpha}, error = {error}".format(
                stage=stage+1, alpha=round(stump_alpha, 3), error=round(stump_e, 3)))

        assert len(self.stumps) == len(self.alphas)

        # plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.5,
        #             linewidths=0.5, edgecolors="#fff")
        # plt.savefig('Plot_save.png')
        # plt.show()

    def predict(self, X):
        Y_pred = 0
        for stage in range(self.stages):
            Y_pred += self.alphas[stage] * self.stumps[stage].predict(X)

        Y_pred = (1 * np.sign(Y_pred)).astype(int)

        return Y_pred

    def plot(self, X, Y, n):
        plotx_min, plotx_max = X[:, 0].min(), X[:, 0].max()
        ploty_min, ploty_max = X[:, 1].min(), X[:, 1].max()
        xx, yy = np.meshgrid(np.arange(plotx_min, plotx_max, 0.02),
                             np.arange(ploty_min, ploty_max, 0.02))

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, alpha=.5)
        plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.5,
                    linewidths=0.5, edgecolors="#fff")

        if not os.path.isdir('B18CSE050_Plots'):
            os.mkdir('B18CSE050_Plots')
        plt.savefig('B18CSE050_Plots/Plot_{n}.png'.format(n=n))
        plt.show()


def main():

    # Creating dataset
    if os.path.isfile('B18CSE050_Data.csv'):
        print("Using dataset from 'B18CSE050_Data.csv' file ...")
        df = pd.read_csv('B18CSE050_Data.csv')
        X_df = df.iloc[:, 1:-1]
        Y_df = df.iloc[:, -1]
        X = X_df.values
        Y = Y_df.values
    else:
        print("Creating dataset ...")
        X, Y = make_classification(
            n_samples=1000, n_features=2, n_informative=2, n_redundant=0)
        Y = np.where(Y == 0, -1, 1)

        df = pd.DataFrame(X)
        df['Y'] = Y
        df.to_csv('B18CSE050_Data.csv')

        plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.5,
                    linewidths=0.5, edgecolors="#fff")
        plt.savefig("B18CSE050_Plots/samples_plot.png")

    n = int(input("How many stages do you want to run it for? : "))

    # Classifier
    model = Adaboost()
    model.fit(X, Y, stages=n)
    Y_pred = model.predict(X)
    model.plot(X, Y, n)

    # Performance
    model_acc = accuracy_score(Y, Y_pred)*100
    print("Accuracy: {acc}%".format(acc=model_acc))

    if not os.path.isfile('B18CSE050_Plots/result.txt'):
        logf = open('B18CSE050_Plots/result.txt', 'a')
        logf.write("Stages \t Accuracy \n")
    else:
        logf = open('B18CSE050_Plots/result.txt', 'a')

    logf.write("{stages} \t {accuracy} \n".format(
        stages=n, accuracy=model_acc))

    logf.close()


if __name__ == "__main__":
    main()
