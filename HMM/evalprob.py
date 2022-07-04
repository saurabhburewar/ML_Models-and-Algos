import numpy as np


def forward(y, Tm, Em, initialP):
    k = Tm.shape[0]
    t = len(y)

    # Initialize
    alpha = np.zeros((t, k))
    alpha[0, :] = initialP * Em[:, y[0]]

    # Recursion
    for i in range(1, t):
        for j in range(k):
            alpha[i, j] = alpha[i-1].dot(Tm[:, j]) * Em[j, y[i]]

    prob = sum(alpha[:, k-1])

    return prob


def backward(y, Tm, Em, initialP):
    k = Tm.shape[0]
    t = len(y)

    # Initialize
    beta = np.zeros((t, k))
    beta[t-1] = np.ones((k))

    # Recursion
    for i in range(t-2, -1, -1):
        for j in range(k):
            beta[i, j] = (beta[i+1] * Em[:, y[i+1]]).dot(Tm[j, :])

    prob = sum(beta[0] * Em[:, y[0]] * initialP[:])

    return prob


initialP = np.array([1, 0])
y = [0, 1, 1, 0]

Tm = np.array([[0.6, 0.4],
               [0.3, 0.7]])

Em = np.array([[0.7, 0.3],
               [0.4, 0.6]])

for_prob = forward(y, Tm, Em, initialP)
back_prob = backward(y, Tm, Em, initialP)

print("Probability calculated using alpha vectors - ", for_prob,
      "\nProbability calculated using beta vectors - ", back_prob)
