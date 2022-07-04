import numpy as np


class Kalman2d():

    def __init__(self, t, ux, uy, stda, stdx, stdy):

        self.t = t
        self.u = np.matrix([[ux], [uy]])
        self.x = np.matrix([[0], [0], [0], [0]])
        self.stateTrans = np.matrix([[1, 0, self.t, 0],
                                     [0, 1, 0, self.t],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])

        self.contMat = np.matrix([[(self.t**2)/2, 0],
                                  [0, (self.t**2)/2],
                                  [self.t, 0],
                                  [0, self.t]])

        self.transMat = np.matrix([[1, 0, 0, 0],
                                   [0, 1, 0, 0]])

        self.std = (stdx+stdy)/2

        self.Q = stda*np.eye(4)
        self.R = (self.std**2)*np.eye(2)

        self.P = np.eye(self.stateTrans.shape[1])

    def predict(self):

        # Predicted State estimate
        self.x = np.dot(self.stateTrans, self.x) + np.dot(self.contMat, self.u)

        # Predicted estimate covariance
        self.P = np.dot(np.dot(self.stateTrans, self.P),
                        self.stateTrans.T) + self.Q

        return self.x[0:2]

    def update(self, z):

        # Measurement Pre-fit residual
        y = z - np.dot(self.transMat, self.x)

        # Pre-fit residual covariance
        S = np.dot(self.transMat, np.dot(self.P, self.transMat.T)) + self.R

        # Optimal Kalman gain
        K = np.dot(np.dot(self.P, self.transMat.T), np.linalg.inv(S))

        # Updated state estimate
        self.x = np.round(self.x + np.dot(K, y))

        # Updated estimate covariance
        self.P = (
            np.eye(self.transMat.shape[1]) - (K * self.transMat)) * self.P

        # Measurement Post-fit residual
        y1 = z - np.dot(self.transMat, self.x)

        return self.x[0:2]


t = np.arange(0, 100, 1)
real_pos = np.array(0.1 * ((t**2) - t))
real_pos2d = []
for i in real_pos:
    temp = (i + np.random.normal(0, 100), i + np.random.normal(0, 100))
    real_pos2d.append(temp)

preds = []
meas = []

kfilter = Kalman2d(1, 2, 2, 0.25, 1.2, 1.2)

for i in real_pos:

    mea = (kfilter.transMat * i + np.random.normal(0, 50)).item(0)
    meas.append(mea)

    (x, y) = kfilter.predict()
    pred = (x.item(0), y.item(0))
    preds.append(pred)

    kfilter.update(mea)

print(preds)
