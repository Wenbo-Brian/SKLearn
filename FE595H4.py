### Part 1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston

dir(load_boston())
print(load_boston().DESCR)
X = load_boston().data
y = load_boston().target
df = pd.DataFrame(X, columns=load_boston().feature_names)
df.head()
linear_model = LinearRegression()
linear_model.fit(X, y)
coef = linear_model.coef_
coef





### Part 2
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.cluster import KMeans
 
wine = datasets.load_wine()
X = wine.data[:, 0]
y = wine.data[:, 2]
model = KMeans(n_clusters=3)
model.fit(wine.data)
all_predictions = model.predict(wine.data)
plt.scatter(X, y, c=all_predictions)
plt.title('Kmeans Clustering of 3 Groups')
plt.show()

SSE = []  
for k in range(1,9):
    pre = KMeans(n_clusters=k)
    pre.fit(wine.data)
    SSE.append(pre.inertia_)
x = range(1,9)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(x,SSE,'x--')
plt.title('Best K of the Model')
plt.show()

model = KMeans(n_clusters=2)
model.fit(wine.data)
all_predictions = model.predict(wine.data)
plt.scatter(X, y, c=all_predictions)
plt.title('Kmeans Clustering of 2 Groups')
plt.show()


# Part1: The results shows PTRATIO is the most influential factors, followed by B and AGE.
# Part2: Elbow method can be used to determine the optimal K value of Kmeans clustering. Seen from the figure, 3 distinct groups are suitable for cluster, however the Kmeans clustering graph of 2 groups is better than 3 groups.









