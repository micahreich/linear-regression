def prediction(theta, x):
    """
    Makes a linear prediction given input x parameterized by theta

    :param theta: parameters
    :param x: input
    :return: prediction
    """

    return theta[0] + theta[1]*x


def cost_fn(theta, x, y):
    """
    Calculates the error of current prediction

    :param theta: parameters
    :param x: inputs
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
    :param x: input features for training examples
    :param y: labels
    :param theta: parameters
    :param alpha: learning rate
    :param epochs: training iterations
    :return: final theta vector
    """

    m = len(x)

    for i in range(epochs):
        gradient_0 = 0
        gradient_1 = 0

        for j in range(m-1):
            gradient_0 += prediction(theta, x[j]) - y[j]
            gradient_1 += (prediction(theta, x[j]) - y[j])*x[j]

        theta_temp = [0, 0]
        theta_temp[0] = theta[0] - alpha*gradient_0
        theta_temp[1] = theta[1] - alpha*gradient_1

        theta[0] = theta_temp[0]
        theta[1] = theta_temp[1]

    return theta
