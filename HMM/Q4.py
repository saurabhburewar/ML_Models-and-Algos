import numpy as np
from trainhmm import train


initialP1 = np.array([1, 0, 0])
y1 = np.array([0, 0, 1, 1])

Tm1 = np.array([[0.0, 0.5, 0.5],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0]])

Em1 = np.array([[0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5]])


iterations = 100000
new_Tm1, new_Em1 = train(iterations, y1, Tm1, Em1, initialP1)

np.set_printoptions(precision=3, suppress=True)
print("Transition matrix - \n", new_Tm1,
      "\nEmission matrix - \n", new_Em1)
