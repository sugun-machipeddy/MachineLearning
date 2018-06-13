from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
import numpy as np
from scipy import stats

class RandomForest(RandomTree, DecisionTree):

    def __init__(self, max_depth, num_trees):
        self.num_trees = num_trees
        RandomTree.__init__(self, max_depth=max_depth)
        DecisionTree.__init__(self, max_depth=max_depth, stump_class=RandomStumpInfoGain)

    def fit(self, X, y):
        rt = []
        M = self.num_trees
        for n in range(M):
            #rt.append(RandomTree(self.max_depth))
            rt.append(RandomTree.fit(self, X, y))
        self.rt = rt

    def predict(self, X):
        result = []
        M = self.num_trees
        result.append(DecisionTree.predict(self, X))
        print(result)
        return stats.mode(result)