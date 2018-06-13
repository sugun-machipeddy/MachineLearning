
# coding: utf-8

# # CPSC 340 Assignment 6

# In[7]:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from scipy.sparse import csr_matrix as sparse_matrix

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD


# ## Instructions
# rubric={mechanics:5}
# 
# 
# The above points are allocated for following the [homework submission instructions](https://github.ugrad.cs.ubc.ca/CPSC340-2017W-T2/home/blob/master/homework_instructions.md).

# ## Exercise 1: Finding similar items
# 
# For this question we'll be using the [Amazon product data set](http://jmcauley.ucsd.edu/data/amazon/). The author of the data set has asked for the following citations:
# 
# > Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering.
# > R. He, J. McAuley.
# > WWW, 2016.
# > 
# > Image-based recommendations on styles and substitutes.
# > J. McAuley, C. Targett, J. Shi, A. van den Hengel.
# > SIGIR, 2015.
# 
# We will focus on the "Patio, Lawn, and Garden" section. Download the [ratings](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Patio_Lawn_and_Garden.csv) and place the file in the `data` directory with the original filename. Once you do that, the code below should load the data:

# In[2]:

filename = "ratings_Patio_Lawn_and_Garden.csv"

with open(os.path.join("..", "data", filename), "rb") as f:
    ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
ratings.head()


# We'd also like to construct the user-product matrix `X`. Let's see how big it would be:

# In[3]:

def get_stats(ratings, item_key="item", user_key="user"):
    print("Number of ratings:", len(ratings))
    print("The average rating:", np.mean(ratings["rating"]))

    d = len(set(ratings[item_key]))
    n = len(set(ratings[user_key]))
    print("Number of users:", n)
    print("Number of items:", d)
    print("Fraction nonzero:", len(ratings)/(n*d))
    print("Size of full X matrix: %.2f GB" % ((n*d)*8/1e9))

    return n,d

n,d = get_stats(ratings)


# 600 GB! That is way too big. We don't want to create that matrix. On the other hand, we see that we only have about 1 million ratings, which would be around 8 MB ($10^6$ numbers $\times$ at 8 bytes per double precision floating point number). Much more manageable. 

# In[4]:

def create_X(ratings,n,d,user_key="user",item_key="item"):
    user_mapper = dict(zip(np.unique(ratings[user_key]), list(range(n))))
    item_mapper = dict(zip(np.unique(ratings[item_key]), list(range(d))))

    user_inverse_mapper = dict(zip(list(range(n)), np.unique(ratings[user_key])))
    item_inverse_mapper = dict(zip(list(range(d)), np.unique(ratings[item_key])))

    user_ind = [user_mapper[i] for i in ratings[user_key]]
    item_ind = [item_mapper[i] for i in ratings[item_key]]

    X = sparse_matrix((ratings["rating"], (user_ind, item_ind)), shape=(n,d))
    
    return X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind

X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = create_X(ratings, n, d)


# In[5]:

# sanity check
print(X.shape) # should be number of users by number of items
print(X.nnz)   # number of nonzero elements -- should equal number of ratings


# In[6]:

X.data.nbytes


# (Above: verifying our estimate of 8 MB to store sparse `X`)

# ### 1.1
# rubric={reasoning:2}
# 
# Find the following items:
# 
# 1. the item with the most reviews
# 2. the item with the most total stars
# 3. the item with the highest average stars
# 
# Then, find the names of these items by looking them up with the url https://www.amazon.com/dp/ITEM_ID, where `ITEM_ID` is the id of the item.

# In[11]:

url_amazon = "https://www.amazon.com/dp/%s"

# example:
print(url_amazon % 'B00CFM0P7Y')


# In[1]:

### YOUR CODE HERE ###
totalReviews = X.getnnz(axis = 0)
mostReviews = np.argmax(totalReviews)
print(X.getnnz(axis = 0))
print ('item with most reviews =', mostReviews)
print('item ID = ', item_inverse_mapper[mostReviews])
totalStarsTemp = X.sum(axis = 0)
totalStars = np.array(totalStarsTemp)
mostTotalStars = np.argmax(X.sum(axis = 0))
print(X.sum(axis=0))
print ('item with most total stars', mostTotalStars)
print('item ID = ', item_inverse_mapper[mostTotalStars])
averageStars = totalStars[0]/totalReviews
print(np.max(averageStars))
print (averageStars)
highestAverageStars = np.argmax(averageStars)
print ('item with highest average stars', highestAverageStars)
print ('item ID =', item_inverse_mapper[highestAverageStars])


# ### 1.2
# rubric={reasoning:2}

# Make the following histograms 
# 
# 1. The number of ratings per user
# 2. The number of ratings per item
# 3. The ratings themselves
# 
# For the first two, use
# ```
# plt.yscale('log', nonposy='clip')
# ``` 
# to put the histograms on a log-scale.

# In[2]:

### YOUR CODE HERE ###
ratingPerUser = X.getnnz(axis = 1)
plt.hist(ratingPerUser)
plt.yscale('log', nonposy = 'clip')
plt.title('ratings per users ')
plt.xlabel('ratings')
plt.ylabel('frequency')
plt.savefig('histogram for ratings per user')

ratingsPerItem = X.getnnz(axis=0)
plt.hist(ratingsPerItem)
plt.yscale('log', nonposy = 'clip')
plt.title('ratings per item')
plt.xlabel('ratings')
plt.ylabel('frequency')
plt.savefig('histogram for ratings per item')

plt.hist(ratings['rating'],bins=[1,2,3,4,5,6])
plt.xlabel('Rating')
plt.ylabel('frequency')
plt.savefig('Total Ratings')

# ### 1.3
# rubric={reasoning:1}
# 
# Use scikit-learn's [NearestNeighbors](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html) object (which uses Euclidean distance by default) to find the 5 items most similar to [Brass Grill Brush 18 Inch Heavy Duty and Extra Strong, Solid Oak Handle](https://www.amazon.com/dp/B00CFM0P7Y). 
# 
# The code block below grabs the row of `X` associated with the grill brush. The mappers take care of going back and forther between the IDs (like `B00CFM0P7Y`) and the indices of the sparse array (0,1,2,...).
# 
# Note: keep in mind that `NearestNeighbors` is for taking neighbors across rows, but here we're working across columns.

# In[20]:

grill_brush = "B00CFM0P7Y"
grill_brush_ind = item_mapper[grill_brush]
grill_brush_vec = X[:,grill_brush_ind]

print(url_amazon % grill_brush)


# In[3]:

### YOUR CODE HERE ###
grill_brush = "B00CFM0P7Y"
grill_brush_ind = item_mapper[grill_brush]
grill_brush_vec = X[:,grill_brush_ind]
nn = NearestNeighbors(6)
nn.fit(X.T)
distances, indices = nn.kneighbors(grill_brush_vec.T, return_distance=True)
print ('distances =', distances)
print ('indices =', indices)
for i in range(6):
    print(i,item_inverse_mapper[indices[0][i]])


indices = [[ 93652 103866 103865  98897  72226 102810]]
0 B00CFM0P7Y
1 B00IJB5MCS
2 B00IJB4MLA
3 B00EXE4O42
4 B00743MZCM
5 B00HVXQY9A


# ### 1.4
# rubric={reasoning:1}
# 
# Using cosine similarity instead of Euclidean distance in `NearestNeighbors`, find the 5 products most similar to `B00CFM0P7Y`.

# In[4]:

### YOUR CODE HERE ###
grill_brush = "B00CFM0P7Y"
grill_brush_ind = item_mapper[grill_brush]
grill_brush_vec = X[:,grill_brush_ind]
nn = NearestNeighbors(n_neighbors=6,metric='cosine')
nn.fit(X.T)
distances, indices = nn.kneighbors(grill_brush_vec.T, return_distance=True)
print ('distances =', distances)
print ('indices =', indices)
for i in range(6):
    print(i,item_inverse_mapper[indices[0][i]])

indices = [[ 93652 103866 103867 103865  98068  98066]]
0 B00CFM0P7Y
1 B00IJB5MCS
2 B00IJB8F3G
3 B00IJB4MLA
4 B00EF45AHU
5 B00EF3YF0Y


# ### 1.5
# rubric={reasoning:2}
# 
# For each of the two metrics, compute the compute the total popularity (total stars) of each of the 5 items and report it. Do the results make sense given what we discussed in class about Euclidean distance vs. cosine similarity? 
Total stars for Cosine
1 B00IJB5MCS
Total Stars for  1 266.0
2 B00IJB8F3G
Total Stars for  2 438.0
3 B00IJB4MLA
Total Stars for  3 205.0
4 B00EF45AHU
Total Stars for  4 311.0
5 B00EF3YF0Y
Total Stars for  5 513.0

Total stars for Euclidean
1 B00IJB5MCS
Total Stars for  1 266.0
2 B00IJB4MLA
Total Stars for  2 205.0
3 B00EXE4O42
Total Stars for  3 5.0
4 B00743MZCM
Total Stars for  4 5.0
5 B00HVXQY9A
Total Stars for 5 5.0

# In[5]:

### YOUR CODE HERE ###
grill_brush = "B00CFM0P7Y"
grill_brush_ind = item_mapper[grill_brush]
grill_brush_vec = X[:, grill_brush_ind]
nn = NearestNeighbors(n_neighbors=6)
nn.fit(X.T)
distances, indices = nn.kneighbors(grill_brush_vec.T, return_distance=True)
print ('distances =', distances)
print ('indices =', indices)
totalStars = X.sum(axis=0)
print (totalStars)
for i in range(6):
    temp = item_inverse_mapper[indices[0][i]]
    print(i, temp)
    print('Total Stars for ', i, totalStars[0, indices[0][i]])

# ### 1.6
# rubric={reasoning:3}
# 
# PCA gives us an approximation $X \approx ZW$ where the rows of $Z$ contain a length-$k$ latent feature vectors for each user and the columns of $W$ contain a length-$k$ latent feature vectors for each item.
# 
# Another strategy for finding similar items is to run PCA and then search for nearest neighbours with Euclidean distance in the latent feature space, which is hopefully more meaningful than the original "user rating space". In other words, we run nearest neighbors on the columns of $W$. Using $k=10$ and scikit-learn's [TruncatedSVD](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) to perform the dimensionality reduction, find the 5 nearest neighbours to the grill brush using this method. You can access $W$ via the `components_` field of the `TruncatedSVD` object, after you fit it to the data. 
# 
# Briefly comment on your results.
# 
# Implementation note: when you call on `NearestNeighbors.kneighbors`, it expects the input to be a 2D array. There's some weirdness here because `X` is a scipy sparse matrix but your `W` will be a dense matrix, and they behave differently in subtle ways. If you get an error like "Expected 2D array, got 1D array instead" then this is your problem: a column of `W` is technically a 1D array but a column of `X` has dimension $1\times n$, which is technically a 2D array. You can take a 1D numpy array and add an extra first dimension to it with `array[None]`.
# 
# Conceptual note 1: We are using the "truncated" rather than full SVD since a full SVD would involve dense $d\times d$ matrices, which we've already established are too big to deal with. And then we'd only use the first $k$ rows of it anyway. So a full SVD would be both impossible and pointless.
# 
# Conceptual note 2: as discussed in class, there is a problem here, which is that we're not ignoring the missing entries. You could get around this by optimizing the PCA objective with gradient descent, say using `findMin` from previous assignments. But we're just going to ignore that for now, as the assignment seems long enough as it is (or at least it's hard for me to judge how long it will take because it's new).

# In[6]:

### YOUR CODE HERE ###
svd = TruncatedSVD(n_components = 10)
svd.fit(X)
w = svd.components_
grill_brush = "B00CFM0P7Y"
grill_brush_ind = item_mapper[grill_brush]
grill_brush_vec = w[:,grill_brush_ind]
nn = NearestNeighbors(n_neighbors=6)
nn.fit(w.T)
distances, indices = nn.kneighbors(grill_brush_vec[:,None].T, return_distance=True)
for i in range(6):
       print(i,item_inverse_mapper[indices[0][i]])

1 B000H1SJ8C
2 B001VNC3Q4
3 B000MVLB8W
4 B000X9BNG8
5 B001H1NG1Q


# ## Exercise 2: putting it all together in a CPSC 340 "mini-project"
# rubric={reasoning:25}
# 
# In this open-ended mini-project, you'll explore the [UCI default of credit card clients data set](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). There are 30,000 examples and 24 features, and the goal is to estimate whether a person will default (fail to pay) their credit card bills; this column is labeled "default payment next month" in the data. The rest of the columns can be used as features. 
# 
# 
# 
# **Your tasks:**
# 
# 1. Download the data set and load it in. Since the data comes as an MS Excel file, I suggest using [`pandas.read_excel`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_excel.html) to read it in. See [Lecture 2](https://github.ugrad.cs.ubc.ca/CPSC340-2017W-T2/home/blob/master/lectures/L2.ipynb) for an example of using pandas.
# 2. Perform exploratory data analysis on the data set. Include at least two summary statistics and two visualizations that you find useful, and accompany each one with a sentence explaining it.
# 3. Randomly split the data into train, validation, test sets. The validation set will be used for your experiments. The test set should be saved until the end, to make sure you didn't overfit on the validation set. You are welcome to use scikit-learn's [train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html), which takes care of both shuffling and splitting. 
# 4. Try scikit-learn's [DummyClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) as a baseline model.
# 5. Try logistic regression as a first real attempt. Make a plot of train/validation error vs. regularization strength. What’s the lowest validation error you can get?
# 6. Explore the features, which are described on the UCI site. Explore preprocessing the features, in terms of transforming non-numerical variables, feature scaling, change of basis, etc. Did this improve your results?
# 7. Try 3 other models aside from logistic regression, at least one of which is a neural network. Can you beat logistic regression? (For the neural net(s), the simplest choice would probably be to use scikit-learn's [MLPClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html), but you are welcome to use any software you wish. )
# 8. Make some attempts to optimize hyperparameters for the models you've tried and summarize your results. In at least one case you should be optimizing multiple hyperparameters for a single model. I won't make it a strict requirement, but I recommend checking out one of the following (the first two are simple scikit-learn tools, the latter two are much more sophisticated algorithms and require installing new packages): 
#   - [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)   
#   - [RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
#   - [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn)
#   - [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize)
# 9. Explore feature selection for this problem. What are some particularly relevant and irrelevant features? Can you improve on your original logistic regression model if you first remove some irrelevant features?
# 10. Take your best model overall. Train it on the combined train/validation set and run it on the test set once. Does the test error agree fairly well with the validation error from before? Do you think you’ve had issues with optimization bias? Report your final test error directly in your README.md file as well as in your report.
# 
# **Submission format:**
# Your submission should take the form of a "report" that includes both code and an explanation of what you've done. You don't have to include everything you ever tried - it's fine just to have your final code - but it should be reproducible. For example, if you chose your hyperparameters based on some hyperparameter optimization experiment, you should leave in the code for that experiment so that someone else could re-run it and obtain the same hyperparameters, rather than mysteriously just setting the hyperparameters to some (carefully chosen) values in your code.
# 
# **Assessment:**
# We plan to grade and fairly leniently. We don't have some secret target accuracy that you need to achieve to get a good grade. You'll be assessed on demonstration of mastery of course topics, clear presentation, and the quality of your analysis and results. For example, if you write something like, "And then I noticed the model was overfitting, so I decided to stop using regularization" - then, well, that's not good. If you just have a bunch of code and no text or figures, that's not good. If you do a bunch of sane things and get a lower accuracy than your friend, don't sweat it.
# 
# **And...**
# This style of this "project" question is different from other assignments. It'll be up to you to decide when you're "done" -- in fact, this is one of the hardest parts of real projects. But please don't spend WAY too much time on this... perhaps "a few hours" (2-6 hours???) is a good guideline for a typical submission. Of course if you're having fun you're welcome to spend as much time as you want! But, if so, don't do it out of perfectionism... do it because you're learning and enjoying it.
# 
# 

# In[ ]:

# YOUR CODE AND REPORT HERE, IN A SENSIBLE FORMAT


# ## Exercise 3: Very short answer questions
# rubric={reasoning:7}
# 
# 1. Why is it difficult for a standard collaborative filtering model to make good predictions for new items?
# 2. Consider a fully connected neural network with layer sizes (10,20,20,5); that is, the input dimensionality is 10, there are two hidden layers each of size 20, and the output dimensionality is 5. How many parameters does the network have, including biases?
# 3. Why do we need nonlinear activation functions in neural networks?
# 4. Assuming we could globally minimize the neural network objectve, how does the depth of a neural network affect the fundamental trade-off?
# 5. List 3 forms of regularization we use to prevent overfitting in neural networks.
# 6. Assuming we could globally minimize the neural network objectve, how would the size of the filters in a convolutational neural network affect the fundamental trade-off?
# 7. Why do people say convolutional neural networks just a special case of a fully-connected (regular) neural networks? What does this imply about the number of learned parameters?
# 

# _answer here_
