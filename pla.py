import matplotlib.pyplot as plt
import numpy as np

np.random.seed(567)

# binary classification

# number of points per class
N = 100

X1_1 = np.random.normal(loc=1.0, size=N, scale=0.5)
X2_1 = 2 * X1_1 + np.random.normal(loc=0.0, size=N, scale=0.2) + 2
Y_1 = np.ones(N)

X1_2 = np.random.normal(loc=1.2, size=N, scale=0.5)
X2_2 = np.random.normal(loc=1.2, size=N, scale=0.5)
Y_2 = -np.ones(N)

w = np.random.uniform(low=-1, size=3)

X0 = np.ones(N * 2)

X1 = np.concatenate((X1_1, X1_2))
X2 = np.concatenate((X2_1, X2_2))

X = np.stack((X0, X1, X2), axis=1)
Y = np.concatenate((Y_1, Y_2))

# set errors
num_errors = 0

# number of epochs
num_epochs = 0

while True:
    num_epochs += 1

    for i in range(Y.shape[0]):
        x = X[i, :]
        y = Y[i]

        if np.dot(w, x) * y < 0:
            num_errors += 1
            w += x * y

    if num_errors == 0:  # or num_epochs == 100:
        break

    num_errors = 0

plt.scatter(X1_1, X2_1)
plt.scatter(X1_2, X2_2)
plt.plot([np.min(X2_2), np.max(X2_2)], [np.min(X2_2) * (-w[1] / w[2]) + (-w[0] / w[2]),
                                        np.max(X2_2) * (-w[1] / w[2]) + (-w[0] / w[2])])
print(num_epochs)
plt.show()
