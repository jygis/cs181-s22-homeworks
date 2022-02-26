import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):

        pi_k = np.zeros(shape = (3,1))

        mu_k = np.zeros(shape = (3,2))

        for i in range(3):
            pi_k[i,:] = 1/3

            #pi_k[i,:] = np.mean(y == i)

            ind = np.where(y == i)

            mu_k[i,:] = np.mean(X[ind,:], axis = 1)


        C_k = []

        C = np.zeros(shape = (2,2))

        if self.is_shared_covariance:

            for j in range(3):
                for i in range(len(X)):
                    sub = np.dot((np.mat(X[i,:]) - np.mat(mu_k[j,:])).T, (np.mat(X[i,:]) - np.mat(mu_k[j,:])))
                    C += sub /X.shape[0]
            C_k = [C, C, C]

        else:

            for j in range(3):

                for i in np.where(y == j)[0]:
                    sub = np.dot((np.mat(X[i,:]) - np.mat(mu_k[j,:])).T, (np.mat(X[i,:]) - np.mat(mu_k[j,:])))
                    C += sub
                C_k.append(C / np.sum(y == j))

        self.mu_hat = mu_k
        self.pi_hat = pi_k
        self.C_hat = C_k



        return

    # TODO: Implement this method!
    def predict(self, X_pred):
        pi_p_k = np.zeros(shape = (3, X_pred.shape[0]))
        for i in range(3):
            pi_p_k[i,:] = self.pi_hat[i,:] * mvn.pdf(x = X_pred, mean = self.mu_hat[i,:], cov = self.C_hat[i])

        y_class_pred = np.argmax(pi_p_k, axis = 0)
        return y_class_pred

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        pass
