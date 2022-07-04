import numpy as np
import matplotlib.pyplot as plt


class Kalman():
    def __init__(self, t, u, accStdDev, MeaStdDev):
        self.t = t
        self.u = u
        self.stda = accStdDev
        self.stdm = MeaStdDev

        # State estimate (xk)
        self.x = np.matrix([[0], [0]])

        # State transition matrix (A)
        self.stateTrans = np.matrix([[1, self.t],
                                     [0, 1]])

        # Control input matrix (B)
        self.contMat = np.matrix([[(self.t**2)/2],
                                  [self.t]])

        # Transformation matrix (H)
        self.transMat = np.matrix([[1, 0]])

        # State noise Covariance matrix (Q)
        self.Q = (self.stda**2)*np.eye(2)
        # self.Q = np.matrix([[(self.t**4)/4, (self.t**3)/2],
        #                     [(self.t**3)/2, self.t**2]]) * self.stda**2

        # Measurement Noise Covariance (R)
        self.R = (self.stdm**2)*1

        # Covariance matrix (P)
        self.P = np.eye(self.stateTrans.shape[1])

    def predict(self):

        # Predicted State estimate
        self.x = np.dot(self.stateTrans, self.x) + np.dot(self.contMat, self.u)

        # Predicted estimate covariance
        self.P = np.dot(np.dot(self.stateTrans, self.P),
                        self.stateTrans.T) + self.Q

        return self.x

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


t = np.arange(0, 100, 1)
real_pos = np.array(0.1 * ((t**2) - t))
preds = []
meas = []

kfilter = Kalman(1, 2, 0.25, 1.2)

for i in real_pos:

    mea = (kfilter.transMat * i + np.random.normal(0, 50)).item(0)
    meas.append(mea)

    pred = kfilter.predict()[0]
    preds.append(pred)

    kfilter.update(mea)


plt.title("Q=cI and R=dI")
# plt.title("Calculated Q and R")
plt.xlabel("Time")
plt.ylabel("Position")

x = t
y1 = meas
y2 = real_pos
y3 = np.squeeze(preds)

plt.plot(x, y1, label="Measurements")
plt.plot(x, y2, label="Actual positions")
plt.plot(x, y3, label="Predictions")

plt.legend()
plt.savefig("Q=cIAndR=dI_1.png")
# plt.savefig("CalculatedQR_1.png")
plt.show()
