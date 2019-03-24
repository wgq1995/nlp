import keras.backend as k
import numpy as np

a = np.array([[1, 2, 3]])
b = np.array([[1, 1, 1]])
b = np.reshape(b, (3, 1))

a = k.variable(value=a)
b = k.variable(value=b)
sess = k.get_session()

print(a.eval(session=sess)) #  output: array([[ 1.,  2.,  3.]], dtype=float32)

print(k.dot(a, b).eval(session=sess)) #  output: array([[ 6.]], dtype=float32)
