    # basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.pyplot as plt

# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from skimage.io import imread, imshow, imsave


# our code
from naive_bayes import NaiveBayes

from decision_stump import DecisionStump, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest # TODO

from kmeans import Kmeans
from kmedians import Kmedians # TODO
from quantize_image import ImageQuantizer
from sklearn.cluster import DBSCAN

def plot_2dclustering(X,y):
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.title('Cluster Plot')


def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == '1.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]
        print(len(wordlist))
        print(wordlist[50])
        print(len(X))
        x_i = X[500]
        print(np.where(x_i))#gives which indices have positive values
        print(wordlist[5],wordlist[25], wordlist[35],wordlist[92])
        print(y[500])
        print(groupnames)
        

    elif question == '1.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = RandomForestClassifier()
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Random Forest (sklearn) validation error: %.3f" % v_error)

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (ours) validation error: %.3f" % v_error)

        model = BernoulliNB()
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (sklearn) validation error: %.3f" % v_error)

    elif question == '2':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)


        print("Our implementations:")
        print("  Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
        print("  Random tree info gain")
        evaluate_model(RandomTree(max_depth=np.inf))
        print("  Random forest info gain")
        #raise NotImplementedError()
        evaluate_model(RandomForest(max_depth=np.inf, num_trees=50)) # TODO: implement this

        print("sklearn implementations")
        print("  Decision tree classifier info gain")
        evaluate_model(DecisionTreeClassifier(criterion="entropy"))
        print("  Random forest info gain")
        evaluate_model(RandomForestClassifier(criterion="entropy"))
        print("  Random forest info classifier  gain, more trees")
        evaluate_model(RandomForestClassifier(criterion="entropy", n_estimators=50))


    elif question == '3':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        print(model.predict(X))
        model.error(X)
        #print(X)
        plot_2dclustering(X, model.predict(X))

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '3.1':
        X = load_dataset('clusterData.pkl')['X']
        N, D = X.shape
        print('N =', N)
        print('D =', D)

        for n in range(50):
            model = Kmeans(k=4)
            model.fit(X)
            model.error(X) #error calls the predict function
            plot_2dclustering(X, model.predict(X))

            fname = os.path.join("..", "figs", "kmeans_error")
            name = fname + str(n) + ".png"
            plt.savefig(name)


    elif question == '3.2':
        X = load_dataset('clusterData.pkl')['X']
        s = np.zeros(50)
        errort = np.zeros(50)
        for n in range(50):
            M = []
            for i in range(1, 11):
                M.append(i)

            print(M)
            k = random.choice(M)
            s[n] = k
            print('n=',n)
            print('k =', k)
            model = Kmeans(k)
            model.fit(X)
            errort[n] = model.error(X)  # error calls the predict function
            plot_2dclustering(X, model.predict(X))

            fname = os.path.join("..", "figs", "kmeans_error")
            name = fname + str(n) + ".png"
            plt.savefig(name)

        plt.plot(s, errort, label="elbow_method")
        plt.xlabel("k")
        plt.ylabel("error")
        plt.legend()
        fname = os.path.join("..", "figs", "q3.3_4_tree_errors.pdf")
        plt.savefig(fname)

    elif question == '3.3':
        X = load_dataset('clusterData2.pkl')['X']
        s = np.zeros(50)
        errort = np.zeros(50)
        for n in range(50):
            M = []
            for i in range(1, 11):
                M.append(i)

            print(M)
            k = random.choice(M)
            s[n] = k
            print('n=', n)
            print('k =', k)
            model = Kmeans(k)
            model.fit(X)
            errort[n] = model.error(X)  # error calls the predict function
            plot_2dclustering(X, model.predict(X))

            fname = os.path.join("..", "figs", "kmeans_3.3")
            name = fname + str(n) + ".png"
            plt.savefig(name)

        plt.plot(s, errort, label="elbow_method")
        plt.xlabel("k")
        plt.ylabel("error")
        plt.legend()
        fname = os.path.join("..", "figs", "q3.3.2.errors.pdf")
        plt.savefig(fname)

        s1 = np.zeros(50)
        errort1 = np.zeros(50)
        for n in range(50):
            M = []
            for i in range(1, 11):
                M.append(i)

            print(M)
            k = random.choice(M)
            s1[n] = k
            print('n=', n)
            print('k =', k)
            model = Kmedians(k=4)
            model.fit(X)
            #errort1[n] =
            model.error(X)  # error calls the predict function
            plot_2dclustering(X, model.predict(X))

            fname = os.path.join("..", "figs", "kmedians_3.3.3")
            name = fname + str(n) + ".png"
            plt.savefig(name)

        plt.plot(s1, errort1, label="elbow_method")
        plt.xlabel("k")
        plt.ylabel("error")
        plt.legend()
        fname = os.path.join("..", "figs", "q3.3.4.medians_errors.pdf")
        plt.savefig(fname)



    elif question == '3.4':
        X = load_dataset('clusterData2.pkl')['X']

        model = DBSCAN(eps=13, min_samples=3)
        y = model.fit_predict(X)

        print("Labels (-1 is unassigned):", np.unique(model.labels_))

        plot_2dclustering(X,y)
        fname = os.path.join("..", "figs", "clusterdata_dbscan_2clusters.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == '4':
        img = imread(os.path.join("..", "data", "mandrill.jpg"))

        # part 1: implement quantize_image.py
        # part 2: use it on the doge
        for b in [1,2,4,6]:
            quantizer = ImageQuantizer(b)
            q_img = quantizer.quantize(img)
            d_img = quantizer.dequantize(q_img)

            plt.figure()
            plt.imshow(d_img)
            fname = os.path.join("..", "figs", "b_{}_image.png".format(b))
            plt.savefig(fname)
            print("Figure saved as '%s'" % fname)

            plt.figure()
            plt.imshow(quantizer.colours[None] if b/2!=b//2 else np.reshape(quantizer.colours, (2**(b//2),2**(b//2),3)))
            plt.title("Colours learned")
            plt.xticks([])
            plt.yticks([])
            fname = os.path.join("..", "figs", "b_{}_colours.png".format(b))
            plt.savefig(fname)
            print("Figure saved as '%s'" % fname)


    else:
        print("Unknown question: %s" % question)
