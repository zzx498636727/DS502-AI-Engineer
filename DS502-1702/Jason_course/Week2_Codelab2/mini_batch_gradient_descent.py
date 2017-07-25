import pandas as pd
import numpy as np
import matplotlib.pyplot as mtplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer

def load_data(file, feature_list):
    df = pd.read_csv(file, names = feature_list)
    df_array = df.values
    feature = df_array[:, :-1]
    result = df_array[:, -1]
    feature_normalized = data_normalizer(norm_type='l2', data=feature)
    return feature_normalized, result

def data_normalizer(data, norm_type):
    """
    :param norm_type: 'l1','l2', 'min_max'
    :param data
    :return: data_normalized
    """
    if norm_type == 'l2' or norm_type == 'l1':
        normalizer = Normalizer(norm = norm_type)
        data_nomalized = normalizer.fit_transform(data)

    if norm_type == 'min_max':
        normalizer = MinMaxScaler(feature_range=(0, 1))
        data_nomalized = normalizer.fit_transform(data)

    return data_nomalized

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def h(x, w):
    return sigmoid(np.dot(np.c_[np.ones(x.shape[0]), x], w.T)).reshape(x.shape[0], )

def logistic_gradient_step(x, y, w):
    y_hat = h(x, w)
    loss = - (1.0 / x.shape[0]) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    gradient = np.dot((y_hat - y), np.c_[np.ones(x.shape[0]), x])
    return loss, gradient

def logistic_regression(x, y, w, num_epoch, learning_rate, batch_size, epsilon, momentum = None, adagrad = False):
    n = x.shape[0]
    p = x.shape[1]
    loss_delta = float("inf")
    loss_history = [1.0]
    momentum_vector = [0.0]
    gradient_history = [np.ones(p + 1)]
    i = 0
    while i < num_epoch and loss_delta > epsilon:
        for j in range(0, n, batch_size):
            x_batch = x[j : batch_size + j, :]
            y_batch = y[j : batch_size + j]
            loss_step, gradient_step = logistic_gradient_step(x_batch, y_batch, w)
            if momentum is not None:
                gradient_delta = momentum * momentum_vector[-1] + learning_rate * gradient_step
                momentum_vector.append(gradient_delta)
            elif adagrad is True:
                gradient_delta = learning_rate * 1.0 * gradient_step / np.sqrt(np.sum(gradient_history, axis=0))
                gradient_history.append(gradient_delta)
            else:
                gradient_delta = learning_rate * gradient_step
            w -= gradient_delta
            loss_delta = abs(loss_step - loss_history[-1])
            loss_history.append(loss_step)
            print('Epoch [{}]: loss={}; loss_delta: {}'.format(i, loss_step, loss_delta))
            if loss_delta < epsilon:
                print('Stop @: Epoch [{}]: loss={}; loss_delta: {}'.format(i, loss_step, loss_delta))
                break
        i += 1
    return np.array(loss_history), w

def predict(x, w):
    predict_prob = sigmoid(np.dot(w, np.c_[np.ones(x.shape[0]), x].T))
    predict_value = np.where(predict_prob > 0.5, 1, 0)
    return predict_value, predict_prob

def validation_printer(x_test, y_test, w, gd_type):
    y_predict, y_prob = predict(x_test, w)
    accuracy = np.sum(y_predict == y_test) * 1.0 / x_test.shape[0]

    print("Gradient descent type {}:".format(gd_type))
    print("Weights: {}".format(w))
    print("Accuracy: {}".format(accuracy))
    print("")

if __name__ == '__main__':
    feature_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    file = './pima-indians-diabetes.data.csv'
    x, y = load_data(file, feature_names)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    # Parameter initialization
    num_epoch = 10
    epsilon = 1e-5
    learning_rate = 0.01
    train_loss = []
    batch_size = 20

    # Weight initialization
    np.random.seed(0)
    weights = np.random.rand(1, x_train.shape[1] + 1)

    # Mini Batch GD
    train_loss_MiniBatchGD, weights_MiniBatchGD = logistic_regression(x_train, y_train, weights,
                                                 num_epoch, learning_rate, batch_size,
                                                 epsilon)
    validation_printer(x_test, y_test, weights_MiniBatchGD, "Mini Batch GD")

    # Mini Batch GD with Momentum
    train_loss_MiniBatchGDMom, weights_MiniBatchGDMom = logistic_regression(x_train, y_train, weights,
                                                                      num_epoch, learning_rate, batch_size,
                                                                      epsilon, momentum=0.9)
    validation_printer(x_test, y_test, weights_MiniBatchGDMom, "Mini Batch GD Momentum")

    # Mini Batch GD with Adagrad
    train_loss_MiniBatchGDAdagrad, weights_MiniBatchGDAdagrad = logistic_regression(x_train, y_train, weights,
                                                                            num_epoch, learning_rate, batch_size,
                                                                            epsilon, adagrad=True)
    validation_printer(x_test, y_test, weights_MiniBatchGDAdagrad, "Mini Batch GD Adagrad")

    #  Plot loss
    mtplot.show()
    mtplot.plot(range(len(train_loss_MiniBatchGD)), train_loss_MiniBatchGD[0:], color = 'red', label = "Mini Batch GD")
    mtplot.plot(range(len(train_loss_MiniBatchGDMom)), train_loss_MiniBatchGDMom[0:], color='blue', label="Mini Batch GD with Momentum")
    mtplot.plot(range(len(train_loss_MiniBatchGDAdagrad)), train_loss_MiniBatchGDAdagrad[0:], color='green', label="Mini Batch with Adagrad")







