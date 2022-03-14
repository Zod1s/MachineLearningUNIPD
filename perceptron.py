# First example of perceptron

# with x0
dataset = [[1, 3, 1, -1], [1, 1, 2, 1], [1, 2.5, 2, -1], [1, 1, 4, -1]]

# initialization of vector of weights
w = [0.1, 0.1, 0.5]

# hyperparameters
num_iter = 50

# number of errors in a cycle
num_errors = 0

# while True:
for _ in range(num_iter):
    # training
    for elem in dataset:
        out = sum([w[i] * elem[i] for i in range(len(w))])
        if (out >= 0 and elem[-1] < 0) or (out < 0 and elem[-1] >= 0):
            # updating weights
            w = [w[i] + elem[i] * elem[-1] for i in range(len(w))]
            num_errors += 1

    print("number of errors:", num_errors)
    if num_errors == 0:
        break
    num_errors = 0

print()
for elem in dataset:
    print(sum([w[i] * elem[i] for i in range(len(w))]))
