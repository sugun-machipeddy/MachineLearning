import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):

        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))
        #print(f)
        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        self.funObj(self.w, X, y)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
        #print(f)
    def predict(self, X):
        return np.sign(X@self.w)

class logRegL2(logReg):
    # L2 regularization
    def __init__(self, verbose=0, lammy=1.0, maxEvals=100):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + (self.lammy/2)*np.sum(w**2)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy*w

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)

class logRegL1(logReg):
    # L2 regularization
    def __init__(self, verbose=0, lammy=1.0, maxEvals=100):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + self.lammy*(np.sum(w))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy *(np.ones(w.shape))

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w,
                                      self.lammy,self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)

class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, lammy=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):

        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + self.lammy*((self.w != 0).sum())
        print(f)
        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj, np.zeros(len(ind)), self.maxEvals, X[:,ind], y, verbose=0)
        selected = set()
        selected.add(0)
        print(list(selected))
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            #print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(1,d): #bias always included
                if i in selected:
                    continue
                else:
                    selected_new = selected | {i}
                # TODO for Q2.3: Fit the model with 'i' added to the features,
                    self.w = np.zeros(d)
                    self.w[list(selected_new)], f = minimize(list(selected_new))

                # then compute the loss and update the minLoss/bestFeature
                    if f <= minLoss:
                        minLoss = f
                        bestFeature = i

            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

class logLinearClassifier(logReg): #inheriting logistic regression

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            #self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)
            self.W[i] = np.zeros(d)
            self.funObj(self.W[i], X, y)
            #utils.check_gradient(self, X, y)
            (self.W[i], f) = findMin.findMin(self.funObj, self.W[i],
                                          self.maxEvals, X, ytmp, verbose=self.verbose)

    def predict(self, X):
        return np.argmax(X @ self.W.T, axis=1)


class softmaxClassifier(logReg):

    def funObj(self, w, X, y):
        n, d = X.shape
        k = np.unique(y).size
        self.W = np.reshape(w, (k, d))
        XW = X@self.W.T
        XW_exp = np.exp(XW)
        # calculate the function value
        f = 0
        for i in range(len(X)):
            f += -XW[i][y[i]] + np.log(np.sum(XW_exp[i]))

        g = np.zeros((k, d))
        pr = np.zeros((n, k))
        for i in range(n):
            for j in range(k):
                pr[i][j] = (np.exp(XW[i][j])/(np.sum(XW_exp[i]))) - (y[i] == j)

        #g = pr.T.dot(X)
        g = pr.T@X
        #print(g.shape)
        return f, g.flatten()

    def fit(self,X, y):
        n, d = X.shape
        k = np.unique(y).size
        # Initial guess
        self.w = np.zeros(k*d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        return np.argmax(X @ self.W.T, axis=1)