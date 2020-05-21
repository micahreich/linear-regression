def prediction(theta, x):
    """
    Makes a linear prediction given input x parameterized by theta

    :param theta: parameters
    :param x: input
    :return: prediction
    """
    hypothesis = 0

    j = len(theta)

    for i in range(j):
        hypothesis += theta[i] * x[i]

    return hypothesis


def cost_fn(theta, x, y):
    """
    Calculates the error of current prediction

    :param theta: parameters
    :param x: inputs of shape
        [[x0, x1, x2, x3],
         [x0, x1, x2, x3]] etc.
    :param y: ground truth labels
    :return: cost
    """

    m = len(x)

    cost = 0
    for i in range(m):
        cost += (prediction(theta, x[i]) - y[i]) ** 2

    cost *= 1/2
    return cost


def gradient_descent(theta, x, y, alpha=0.01, epochs=1500):
    """
    Does gradient descent algorithm to minimize cost function and determine parameters theta
    :param theta: starting parameters
    :param x: input features for training examples
    :param y: labels
    :param theta: parameters
    :param alpha: learning rate
    :param epochs: training iterations
    :return: final theta vector
    """

    m = len(x)

    j = len(x[0])

    for i in range(epochs):
        gradients = [0] * j
        theta_temp = [0] * len(theta)

        for k in range(m-1):
            for p in range(len(gradients)):
                gradients[p] += (prediction(theta, x[k]) - y[k])*x[p]

        for k in range(len(theta_temp)):
            theta_temp[k] = theta[k] - alpha*gradients[k]
            theta[k] = theta_temp[k]

    return theta
