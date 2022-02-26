import numpy as np

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def predict(self, X_pred):

        predictions = []

        for i in range(X_pred.shape[0]):

            distance = []

            for j in range(self.X.shape[0]):

                r = ((X_pred[i,0] - self.X[j,0])/3)**2 + (X_pred[i,1] - self.X[j,1])**2

                distance.append(r)

            NN = np.argsort(distance)[:self.K]

            one_hot_y = np.zeros(shape = (len(X), 3))
            for i in range(len(self.y)):
                one_hot_y[i, self.y[i]] = 1

            y_NN = one_hot_y[NN,:]

            y_class_pred = np.argmax(np.sum(y_NN, axis = 0))

            predictions.append(y_class_pred)

        return np.array(predictions)

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y
