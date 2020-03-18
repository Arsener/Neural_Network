import numpy as np


class MLPClassifier:
    def __init__(self, hidden_layer_sizes, activation='relu', learning_rate=0.001, batch_size=64, max_epoch=40):
        self.hidden_layers_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch

        # 初始化参数
        self.layer_num = len(hidden_layer_sizes)
        self.weights = [None]
        self.bias = [np.ones((1, self.hidden_layers_sizes[0]))]
        pre = self.hidden_layers_sizes[0]
        for i in range(1, self.layer_num):
            self.weights.append(
                2 * (np.random.random(size=(pre, self.hidden_layers_sizes[i])) / pre ** 0.5) - 1. / pre ** .5)
            self.bias.append(np.random.randn(1, self.hidden_layers_sizes[i]))
            pre = self.hidden_layers_sizes[i]

        if activation == 'relu':
            self.act = self.relu
            self.der_act = self.derivation_relu
        elif activation == 'sigmoid':
            self.act = self.sigmoid
            self.der_act = self.derivation_sigmoid
        else:
            raise ValueError("activation function must be 'relu' or 'sigmoid'.")

        # 用于反向求导存中间变量
        self.z = []
        self.delta_w = []
        self.delta_b = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivation_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x):
        return np.maximum(0, x)

    def derivation_relu(self, x):
        x[x < 0] = 0
        x[x > 0] = 1
        return x

    def softmax(self, x):
        e = np.exp(x)
        return e / np.sum(e, axis=1).reshape(x.shape[0], 1)

    def cross_entropy(self, y, y_hat):
        return -np.sum(y * np.log(y_hat + 1e-6)) / y.shape[0]

    def forward(self, X):
        # 将X作为输入层的激活层输出
        self.z = [X]
        for w, b in zip(self.weights[:-1], self.bias[:-1]):
            if len(self.z) == 1:
                self.z.append(np.dot(self.z[-1], w) + b)
            else:
                self.z.append(np.dot(self.act(self.z[-1]), w) + b)

        return self.softmax(np.dot(self.act(self.z[-1]), self.weights[-1]) + self.bias[-1])

    def backward(self, y, y_hat):
        self.delta_b.clear()
        self.delta_w.clear()
        n = y.shape[0]
        delta_z = (y_hat - y) / n
        layer_index = len(self.z) - 1
        while layer_index >= 0:
            self.delta_b.append(np.sum(delta_z, axis=0, keepdims=True))
            if layer_index == 0:
                self.delta_w.append(np.dot(self.z[layer_index].T, delta_z))
            else:
                self.delta_w.append(np.dot(self.act(self.z[layer_index]).T, delta_z))

            # 更新对z的导数
            if layer_index > 0:
                delta_z = np.dot(delta_z, self.weights[layer_index].T) * self.der_act(self.z[layer_index]) / n
            layer_index -= 1

    def fit(self, X, y):
        input_shape = X.shape[1]
        output_shape = y.shape[1]
        self.weights[0] = 2 * (np.random.random(size=(input_shape, self.hidden_layers_sizes[0])) /
                               input_shape ** 0.5) - 1. / input_shape ** .5
        self.weights.append(2 * (np.random.random(size=(self.hidden_layers_sizes[-1], output_shape)) /
                                 self.hidden_layers_sizes[-1] ** 0.5) - 1. / self.hidden_layers_sizes[-1] ** .5)
        self.bias.append(np.random.randn(1, output_shape))

        n_sample = X.shape[0]
        iter_num = n_sample // self.batch_size
        if n_sample % self.batch_size: iter_num += 1

        for epoch in range(self.max_epoch):
            for iter in range(iter_num):
                X_batch = X[iter * self.batch_size: min((iter + 1) * self.batch_size, n_sample)]
                y_batch = y[iter * self.batch_size: min((iter + 1) * self.batch_size, n_sample)]
                y_pred = self.forward(X_batch)
                self.backward(y_batch, y_pred)

                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * self.delta_w[self.layer_num - i]
                    self.bias[i] -= self.learning_rate * self.delta_b[self.layer_num - i]

            y_pred = self.forward(X)
            loss = self.cross_entropy(y, y_pred)
            print('Epoch {}: Loss: {}'.format(epoch, loss))

    def predict_prob(self, X):
        return self.forward(X)

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argwhere(y_pred == np.max(y_pred, axis=1).reshape(y_pred.shape[0], 1))[:, 1:]
