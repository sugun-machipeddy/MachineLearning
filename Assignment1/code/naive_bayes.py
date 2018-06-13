import numpy as np

class NaiveBayes:
    # Naive Bayes implementation.
    # Assumes the feature are binary.
    # Also assumes the labels go from 0,1,...C-1

    def __init__(self, num_classes, beta=1):
        self.num_classes = num_classes
        self.beta = beta

    def fit(self, X, y):
        N, D = X.shape

        beta = self.beta
        # Compute the number of class labels
        C = self.num_classes
        counts = np.bincount(y)
        p_y = counts / N
        p_xy = np.zeros((D, C))
        #print(p_xy)
        p_d = np.zeros(D)
        #
        #p_xy[d,c]=p(x(i,j)=1|y(i)==c)
        for h in range(C):
            for d in range(D):

                p_d[d] = np.bincount(X[:][d])[0]/N
                count = 0
                for n in range(N):
                    if X[n,d] == 1 and y[n] ==1:
                        count += 1

                p_xy[d,h] = ((count/np.bincount(X[:][d])[0])*p_d[d] + beta )/(y[h]*(1+beta))
                #p_xy[d, h] = ((count / np.bincount(X[:][d])[0]) * p_d[d]) / y[h]

        self.p_y = p_y
        self.p_xy = p_xy


    # This function is provided just for your understanding.
    # It should function the same as predict()
    def predict(self, X):

        N, D = X.shape
        C = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(N)
        for n in range(N):

            probs = p_y.copy() # initialize with the p(y) terms
            for d in range(D):
                if X[n, d] != 0:
                    probs *= p_xy[d, :]
                else:
                    probs *= (1-p_xy[d, :])

            y_pred[n] = np.argmax(probs)

        return y_pred
