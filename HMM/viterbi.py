import numpy as np


def viterbi(y, A, B, initialP=None):
    k = A.shape[0]
    t = len(y)
    T1 = np.empty((k, t), 'd')
    T2 = np.empty((k, t), 'B')

    if initialP == None:
        initialP = np.zeros(k)
        initialP[0] = 1

    # initial values in T1 and T2
    for i in range(k):
        T1[i, 0] = initialP[i] * B[i][y[0]]
        T2[i, 0] = 0

    # Update T1 and T2
    for j in range(1, t):
        T1[:, j] = np.max(T1[:, j-1] * A.T * B[np.newaxis, :, y[j]].T, 1)
        T2[:, j] = np.argmax(T1[:, j-1] * A.T, 1)

    x = np.empty(t, 'B')
    x[-1] = np.argmax(T1[:, t-1])

    for j in reversed(range(1, t)):
        x[j-1] = T2[x[j], j]

    return x


Tm = np.array([[0.6, 0.4],
               [0.3, 0.7]])

Em = np.array([[0.7, 0.3],
               [0.4, 0.6]])

initialP = [1, 0]
y = [0, 0, 1, 1]

x = viterbi(y, Tm, Em, initialP)
print("Most likely state sequence: ", x)
