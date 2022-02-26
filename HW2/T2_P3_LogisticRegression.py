import numpy as np



# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

def softmax(x):
    ret = np.zeros(shape = x.shape)
    for i in range(x.shape[0]):
        ret[i,:] = np.exp(x[i,:]) / np.sum(np.exp(x[i,:]))
    return ret

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):

        feature = np.stack([np.ones(len(X)), X[:,0], X[:,1]], axis = 1)

        one_hot_y = np.zeros(shape = (len(X), 3))

        for i in range(len(y)):
            one_hot_y[i, y[i]] = 1

        self.W = np.random.rand(feature.shape[1], 3)

        self.loss = []


        for i in range(200):

            dEw = np.dot(feature.T, (softmax(np.dot(feature, self.W)) - one_hot_y)) + 2 * self.lam * self.W
            self.W -= self.eta * dEw

            ylny = 0

            for k in range(one_hot_y.shape[0]):
                for j in range(one_hot_y.shape[1]):
                    ylny += one_hot_y[k, j] * np.log(softmax(np.dot(feature, self.W))[k, j])
            self.loss.append(-ylny)

        return

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.

        feature = np.stack([np.ones(len(X_pred)), X_pred[:,0], X_pred[:,1]], axis = 1)

        mu = softmax(np.dot(feature, self.W))

        return np.argmax(mu, axis = 1)

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):

        plt.subplot()
        plt.figure()
        plt.plot(self.loss)
        plt.show()
