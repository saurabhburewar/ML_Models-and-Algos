import numpy as np
from trainhmm import train


initialP = np.array([1, 0])
y = np.array([0, 1, 0, 1, 1])

Tm = np.array([[0.96, 0.04],
               [1.0, 0.0]])

Em = np.array([[0.52, 0.48],
               [0.0, 1.0]])

iterations = 100000
new_Tm, new_Em = train(iterations, y, Tm, Em, initialP)

np.set_printoptions(precision=3, suppress=True)
print("Transition matrix - \n", new_Tm,
      "\nEmission matrix - \n", new_Em)
