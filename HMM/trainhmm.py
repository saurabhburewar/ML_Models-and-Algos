import numpy as np
from tqdm import tqdm


def train(itrs, y, Tm, Em, initialP):
    k = Tm.shape[0]
    t = len(y)

    print("Running for {} iterations/epochs".format(itrs))
    for itr in tqdm(range(itrs)):

        # Forward part
        alpha = np.zeros((t, k))
        alpha[0, :] = initialP * Em[:, y[0]]

        for i in range(1, t):
            for j in range(k):
                alpha[i, j] = alpha[i-1].dot(Tm[:, j]) * Em[j, y[i]]

        # Backward part
        beta = np.zeros((t, k))
        beta[t-1] = np.ones((k))

        for i in range(t-2, -1, -1):
            for j in range(k):
                beta[i, j] = (beta[i+1] * Em[:, y[i+1]]).dot(Tm[j, :])

        # E step of the algorithm
        xi = np.zeros((k, k, t-1))
        for i in range(t-1):
            denominator = np.dot(
                np.dot(alpha[i, :].T, Tm) * Em[:, y[i+1]].T, beta[i+1, :])
            for j in range(k):
                numerator = alpha[i, j] * Tm[j, :] * \
                    Em[:, y[i+1]].T * beta[i+1, :].T
                xi[j, :, i] = numerator / denominator

        gamma = np.sum(xi, axis=1)

        # M step of the algorithm
        Tm = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        gamma = np.hstack(
            (gamma, np.sum(xi[:, :, t-2], axis=0).reshape((-1, 1))))

        n = Em.shape[1]
        denominator = np.sum(gamma, axis=1)
        for i in range(n):
            Em[:, i] = np.sum(gamma[:, y == i], axis=1)

        Em = np.divide(Em, denominator.reshape((-1, 1)))

    return Tm, Em
