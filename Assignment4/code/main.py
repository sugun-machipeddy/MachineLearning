import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

import utils
from pca import PCA, AlternativePCA, RobustPCA
from manifold import MDS, ISOMAP

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question
    def frobenius_norm(X):
        n,d = X.shape
        sum = 0
        for i in range(n):
            for j in range(d):
                sum += pow((X[i][j]),2)

        return np.sqrt(sum)

    if question == '2':
        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        f1, f2 = np.random.choice(d, size=2, replace=False)
        #print(X[:,f1])
        #print(f1)
        plt.figure()
        plt.scatter(X[:,f1], X[:,f2])
        plt.xlabel("$x_{%d}$" % f1)
        plt.ylabel("$x_{%d}$" % f2)
        for i in range(n):
            plt.annotate(animals[i], (X[i,f1], X[i,f2]))
        utils.savefig('two_random_features.png')

    elif question == '2.2': 
        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape
        print(n,d)
        # standardize columns
        X = utils.standardize_cols(X)
        #print(X)
        ########
        # TODO #
        model = PCA(k=2)
        model.fit(X)
        z = model.compress(X)
        #print(z)
        plt.figure()
        plt.scatter(z[:, 0], z[:, 1])
        plt.xlabel("z1")
        plt.ylabel("z2")
        for i in range(int(n)):
            plt.annotate(animals[i], (z[i, 0], z[i, 1]))
            #print(animals[i])
        utils.savefig('q2.2_PCA.png')
        ########

    elif question == '2.3':
        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n, d = X.shape
        model = PCA(k=2)
        model.fit(X)
        z= model.compress(X)
        #X_new = model.expand(z)
        #frobenius_norm(X)
        #variance = frobenius_norm(X_new - X)/frobenius_norm(X)
        print (model.variance(X, z))

        for i in range(1,d):
            model = PCA(k = i)
            model.fit(X)
            z = model.compress(X)
            print ('variance for k =', i, model.variance(X, z))




    elif question == '3.1': 
        X = load_dataset('highway.pkl')['X'].astype(float)/255
        n,d = X.shape
        print(n,d)
        h,w = 64,64      # height and width of each image

        k = 5            # number of PCs
        threshold = 0.1  # threshold for being considered "foreground"

        model = AlternativePCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_pca = model.expand(Z)

        model = RobustPCA(k=k) # TODO: implement this class
        model.fit(X)
        Z = model.compress(X)
        Xhat_robust = model.expand(Z)

        fig, ax = plt.subplots(2,3)
        for i in range(10):
            ax[0,0].set_title('$X$')
            ax[0,0].imshow(X[i].reshape(h,w).T, cmap='gray')

            ax[0,1].set_title('$\hat{X}$ (L2)')
            ax[0,1].imshow(Xhat_pca[i].reshape(h,w).T, cmap='gray')
            
            ax[0,2].set_title('$|x_i-\hat{x_i}|$>threshold (L2)')
            ax[0,2].imshow((np.abs(X[i] - Xhat_pca[i])<threshold).reshape(h,w).T, cmap='gray')

            ax[1,0].set_title('$X$')
            ax[1,0].imshow(X[i].reshape(h,w).T, cmap='gray')
            
            ax[1,1].set_title('$\hat{X}$ (L1)')
            ax[1,1].imshow(Xhat_robust[i].reshape(h,w).T, cmap='gray')

            ax[1,2].set_title('$|x_i-\hat{x_i}|$>threshold (L1)')
            ax[1,2].imshow((np.abs(X[i] - Xhat_robust[i])<threshold).reshape(h,w).T, cmap='gray')

            utils.savefig('highway_{:03d}.jpg'.format(i))

    elif question == '4':
        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        model = MDS(n_components=2)
        Z = model.compress(X)

        fig, ax = plt.subplots()
        ax.scatter(Z[:,0], Z[:,1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('MDS')
        for i in range(n):
            ax.annotate(animals[i], (Z[i,0], Z[i,1]))
        utils.savefig('MDS_animals.png')

    elif question == '4.1':
        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        for n_neighbours in [2,3]:
        #for n_neighbours in range(n):
            model = ISOMAP(n_components=2, 
                           n_neighbours=n_neighbours) # TODO: finish implementing this class
            Z = model.compress(X)

            fig, ax = plt.subplots()
            ax.scatter(Z[:,0], Z[:,1])
            plt.ylabel('z2')
            plt.xlabel('z1')
            plt.title('ISOMAP with NN=%d' % n_neighbours)
            for i in range(n):
                ax.annotate(animals[i], (Z[i,0], Z[i,1]))
            utils.savefig('ISOMAP%d_animals.png' % n_neighbours)

    else:
        print("Unknown question: %s" % question)