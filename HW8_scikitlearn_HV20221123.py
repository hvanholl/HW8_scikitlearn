import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
newline = '\n'

data = pd.read_csv('arrhythmia.csv', header = None, na_values = '?')
#print(data.iloc[:, 10:20].head)        # To verify that the ? were replaced

import sklearn
from sklearn.experimental import enable_iterative_imputer       # Needed because the package is experimental
from sklearn.impute import IterativeImputer # The data contains missing values so this will use the dataset to estimate the missing values
from sklearn.preprocessing import scale   # To scale the data prior to k-means
from sklearn.cluster import KMeans        # package to perform k-means analysis
from scipy.spatial.distance import cdist    # Used for finding outliers
import kneed                                # Used for automatically determining elbow for sse
from kneed import KneeLocator

### --------------------------------------- Pre-process the data ------------------ ###

## select only linear data. Removed columns labeled as nominal in the data description file as well as the class variable.
data = data.drop(np.r_[1, 21:27, 33:39, 45:51, 57:63, 69:75, 81:87, 93:99, 105:111, 117:123, 129:135, 141:147, 153:159, 279], 1)
#print(data.iloc[1:10, 190:207])

## Impute the data using all the other data to replace missing values.
imp = IterativeImputer(max_iter = 10, random_state = 3)   # max iterations to limit the computation, defines imputation
imp.fit(data)                           # fits the data to the imputation model
data_imp = imp.transform(data)          # transforms the dataset using the fitted model, replaces missing values 

## Scale the data so that each variable has the same range
x = scale(data_imp)         # scales data so it is centered on zero
# print(x)      # To verify that the scaling worked
# Final k means analysis is run after the sse analysis

### ------------ Sum of Least Squares ---------------------------------------------------- ###

# A list holds the SSE values for each k
sse = []                                # empty list for sum of least squares results
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, random_state = 3)       # iterate through 20 values of k
    kmeans.fit(x)                                       # fit the data
    sse.append(kmeans.inertia_)                         # get the sum of least squares and append to the list
    
plt.style.use("fivethirtyeight")        
plt.plot(range(1, 20), sse)                             # plot the sum of least squares values
plt.xticks(range(1, 20))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title('Sum of Least Squares for All Data')
plt.show()

# To automatically determine the optimal number of clusters
# This finds the 'elbow' of the sse curve
kl = KneeLocator(range(1, 20), sse, curve="convex", direction="decreasing")
print(f'The optimal number of clusters is {kl.elbow}')


kmeans = KMeans(n_clusters=8, random_state=3).fit(x)  # execute the kmeans with 8 clusters because of the elbow results
print(kmeans.fit(x))
#predict the labels of clusters.
label = kmeans.fit_predict(x)
#plotting the results:
plt.scatter(x[:, 0] , x[: , 1], c = label, cmap = 'turbo')
plt.title('K-means Clustering with All Data Included')
plt.show()

# obtaining the centers of the clusters
centroids = kmeans.cluster_centers_
# points array will be used to reach the index easy
points = np.empty((0,len(x[0])), float)
# distances will be used to calculate outliers
distances = np.empty((0,len(x[0])), float)
# getting points and distances
for i, center_elem in enumerate(centroids):
    # cdist is used to calculate the distance between center and other points
    distances = np.append(distances, cdist([center_elem],x[label == i], 'euclidean')) 
    points = np.append(points, x[label == i], axis=0)
percentile = 90
# getting outliers whose distances are greater than some percentile. Choose 90% with some trial and error to remove the data points that were extremely distance from the group.
outliers = points[np.where(distances > np.percentile(distances, percentile))]
no_outliers = points[np.where(distances < np.percentile(distances, percentile))]
print(outliers)
print(no_outliers)
 
## ------------------ Repeat with no_outliers (90TH PERCENTILE)------------------------------###

# A list holds the SSE values for each k
sse = []                                # empty list for sum of least squares results
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, random_state = 3)       # iterate through 15 values of k
    kmeans.fit(no_outliers)                # fit the data
    sse.append(kmeans.inertia_)         # get the sum of least squares and append to the list
    
plt.style.use("fivethirtyeight")        
plt.plot(range(1, 20), sse)             # plot the sum of least squares values
plt.xticks(range(1, 20))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title('Sum of Least Squares for Filtered Data')
plt.show()

# To automatically determine the optimal number of clusters
import kneed
from kneed import KneeLocator
kl = KneeLocator(range(1, 20), sse, curve="convex", direction="decreasing")
print(f'The optimal number of clusters is {kl.elbow}')

kmeans = KMeans(n_clusters=8, random_state=3).fit(no_outliers)  # execute the kmeans with 8 clusters because of the elbow results
# This elbow value is not consistent with the randomness of the analysis, probably because the data is not really clustering at all but 7, 8 , and 9 are the most common results from the elbow method. The curve looks almost linear so it is hard to tell visually.
# I set the random state so that it would be reproducible, but it is pretty variable.
print(kmeans.fit(no_outliers))
#predict the labels of clusters.
label = kmeans.fit_predict(no_outliers)
#plotting the results:
plt.scatter(no_outliers[:, 0] , no_outliers[: , 1], c = label, cmap = 'turbo')
plt.title('K-means Clustering with Outliers Removed (90th Percentile)')
plt.show()


### --------------------------- DISCUSSION -------------------------------- ###

## This dataset doesn't cluster very cleanly with the arrhythmia attributes. There is essentially no visible separation even when the outliers are removed. The results were also highly variable with random events (at both the imputation step and the kmeans steps (for sse and final analysis). I don't know enough about the data or the medical relevance of each attribute to evaluate this critically, but I am guessing that some of the variables are not normally distributed and perhaps the missing values are really throwing things off. Maybe choosing the variables to include more thoughtfully with a field expert would yield more informative results. 