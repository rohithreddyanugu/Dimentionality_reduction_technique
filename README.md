# Dimentionality_reduction_technique

## Principal Component Analysis in Machine Learning:

Principal Component Analysis (PCA) is a statistical techniques used to reduce the dimensionality of the data (reduce the number of features in the dataset) by selecting the most important features that capture maximum information about the dataset.

The features are selected on the basis of variance that they cause in the output. Original features of the dataset are converted to the Principal Components which are the linear combinations of the existing features. The feature that causes highest variance is the first Principal Component. The feature that is responsible for second highest variance is considered the second Principal Component, and so on.

In simple words, Principal Component Analysis is a method of extracting important features (in the form of components) from a large set of features available in a dataset.

PCA finds the directions of maximum variance in high-dimensional data and project it onto a smaller dimensional subspace while retaining most of the information. By projecting our data into a smaller space, we’re reducing the dimensionality of our feature space.

Following are some of the advantages and disadvantages of Principal Component Analysis:

## Advantages of Principal Component Analysis

1. Removes Correlated Features: In a real world scenario, this is very common that you get thousands of features in your dataset. You cannot run your algorithm on all the features as it will reduce the performance of your algorithm and it will not be easy to visualize that many features in any kind of graph. So, you MUST reduce the number of features in your dataset.

You need to find out the correlation among the features (correlated variables). Finding correlation manually in thousands of features is nearly impossible, frustrating and time-consuming. PCA does this for you efficiently.

After implementing the PCA on your dataset, all the Principal Components are independent of one another. There is no correlation among them.

2. Improves Algorithm Performance: With so many features, the performance of your algorithm will drastically degrade. PCA is a very common way to speed up your Machine Learning algorithm by getting rid of correlated variables which don't contribute in any decision making. The training time of the algorithms reduces significantly with less number of features.

So, if the input dimensions are too high, then using PCA to speed up the algorithm is a reasonable choice.

3. Reduces Overfitting: Overfitting mainly occurs when there are too many variables in the dataset. So, PCA helps in overcoming the overfitting issue by reducing the number of features.

4. Improves Visualization: It is very hard to visualize and understand the data in high dimensions. PCA transforms a high dimensional data to low dimensional data (2 dimension) so that it can be visualized easily.

We can use 2D Scree Plot to see which Principal Components result in high variance and have more impact as compared to other Principal Components.

Even the simplest IRIS dataset is 4 dimensional which is hard to visualize. We can use PCA to reduce it to 2 dimension for better visualization.

Consider a situation where we have 50 features (p = 50). There can be p(p-1)/2 scatter plots i.e. 1225 plots possible to analyze the variable relationships. It would be a tedious job to perform exploratory analysis on this data. That is why, we have to use PCA to get rid of this problem.

## Disadvantages of Principal Component Analysis

1. Independent variables become less interpretable: After implementing PCA on the dataset, your original features will turn into Principal Components. Principal Components are the linear combination of your original features. Principal Components are not as readable and interpretable as original features.

2. Data standardization is must before PCA: You must standardize your data before implementing PCA, otherwise PCA will not be able to find the optimal Principal Components.

For instance, if a feature set has data expressed in units of Kilograms, Light years, or Millions, the variance scale is huge in the training set. If PCA is applied on such a feature set, the resultant loadings for features with high variance will also be large. Hence, principal components will be biased towards features with high variance, leading to false results.

Also, for standardization, all the categorical features are required to be converted into numerical features before PCA can be applied.

PCA is affected by scale, so you need to scale the features in your data before applying PCA. Use StandardScaler from Scikit Learn to standardize the dataset features onto unit scale (mean = 0 and standard deviation = 1) which is a requirement for the optimal performance of many Machine Learning algorithms.

3. Information Loss: Although Principal Components try to cover maximum variance among the features in a dataset, if we don't select the number of Principal Components with care, it may miss some information as compared to the original list of features.

## Below is my understanding of the PCA :

1. PCA is one of the way to reduce high dimension features (say 784 in MNIST dataset in our example) to lower dimension without losing the variance of the original data.

Steps to arrive PCA:
lets take our dataset as matrix A

Step 1: Preprocess A i.e standardize the dataset.
Step 2 : Find the co-variance of the A , let's say co-variance matrix as S.
Step 3: Find the eigen values and eigen vectors for the co-variance matrix, S
Step 4 : Find the dot product of the eigen vector and co-variance matrix , S

Eigen Values - Gives us the percentange of variance of the features.
Eigen Vectors - Gives us the direction of the features .

# T-SNE

It is a popular method of exploring high-dimensional data and has become widespread in the world of machine learning. It was introduced by Maaten and Hinton in 2008. It is a very useful dimensionality reduction technique and fairly better than the other techniques mentioned above. It has a very amazing ability to create two-dimensional maps from high dimension data. It’s very easy to implement in python using sci-kit learn.

## How does t-SNE work?
The working of t-SNE requires very non-trivial mathematics and takes a significant effort to understand. What t-SNE roughly does is that it tries to preserve the distances in a neighborhood.
Suppose we are reducing d-dimensional data into 2-dimensional data using t-SNE.
From the above picture we can see that x2 and x3 are in the neighborhood of x1 [N(x1) = {x2, x3}] and x5 is in the neighborhood of x4 [N(x4) = {x5}].
As t-SNE preserves the distances in a neighborhood,
d(x1, x2) ≈ d(x’1, x’2)
d(x1, x3) ≈ d(x’1, x’3)
d(x4, x5) ≈ d(x’4, x’5)
For every point, it constructs a notion of which other points are its ‘neighbors,’ trying to make all points have the same number of neighbors. Then it tries to embed them so that those points all have the same number of neighbors.

## MNIST dataset
MNIST is a computer vision dataset which contains images of the handwritten digits with each image being 28 pixels in height and 28 pixels in width, for a total of 784 pixels.

Now we will be performingPCA and  t-SNE on this dataset with different values of hyperparameters.
With perplexity 30 and n_iter = 1000 and taking only 1000 data points.
