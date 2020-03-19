from MLPClassifier import *


# Load the images and labels from a specified dataset (train or test).
def loadData(which):
    images = np.load("./data/mnist_{}_images.npy".format(which))
    labels = np.load("./data/mnist_{}_labels.npy".format(which))
    return images, labels


def accuracy(Y, y_hat):
    n = Y.shape[0]
    Y_label = np.argwhere(Y == 1)[:, 1:]
    y_hat_label = np.argwhere(y_hat == np.max(y_hat, axis=1).reshape(n, 1))[:, 1:]
    mask = np.zeros(Y_label.shape)
    mask[Y_label == y_hat_label] = 1
    return np.sum(mask) / n


# Load data
trainX, trainY = loadData("train")
testX, testY = loadData("test")

mlp = MLPClassifier((50,), learning_rate=0.05, max_epoch=40, activation='relu')
mlp.fit(trainX, trainY)
y_pred = mlp.predict_prob(testX)

print(accuracy(testY, y_pred))
