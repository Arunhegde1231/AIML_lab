import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def kernel(point, xmat, k):
    m,n = np.shape(xmat)
    weights = np.mat(np.eye(m))
    for j in range(m):
        diff = point - X[j]
        weights[j,j] = np.exp(diff*diff.T/(-2.0*k**2))
    return weights

def localWeight(point, xmat, ymat, k):
    wei = kernel(point,xmat,k)
    return (X.T*(wei*X)).I*(X.T*(wei*ymat.T))
     
def localWeightRegression(xmat, ymat, k):
    m,n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i]*localWeight(xmat[i],xmat,ymat,k)
    return ypred
       
data = pd.read_csv('p9.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)
 
mbill = np.mat(bill)
mtip = np.mat(tip)
m= np.shape(mbill)[1]

one = np.mat(np.ones(m))
X = np.hstack((one.T,mbill.T))

ypred = localWeightRegression(X,mtip,0.5)
SortIndex = X[:,1].argsort(0)
xsort = X[SortIndex][:,0]
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(bill,tip, color='green')
ax.plot(xsort[:,1],ypred[SortIndex], color = 'red', linewidth=5)
plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.show()


'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def kernel(point, xmat, k):
    m, n = np.shape(xmat)
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diff = point - X[j]
        weights[j, j] = np.exp(diff * diff.T / (-2.0 * k ** 2))
    return weights

def localWeight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    return (X.T * (wei * X)).I * (X.T * (wei * ymat.T))

def localWeightRegression(xmat, ymat, k):
    m, n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] * localWeight(xmat[i], xmat, ymat, k)
    return ypred

def GraphPlot(X, ypred):
    SortIndex = X[:, 1].argsort(0)
    xsort = X[SortIndex][:, 0]
    plt.subplot(1, 1, 1)
    plt.scatter(sepal_length, petal_length, color='green')  # Adjusted to use sepal_length and petal_length
    plt.plot(xsort[:, 1], ypred[SortIndex], color='red', linewidth=5)
    plt.xlabel('Sepal Length')  # Adjusted labels
    plt.ylabel('Petal Length')  # Adjusted labels
    plt.show()

# Load Iris dataset
iris = load_iris()
sepal_length = np.array(iris.data[:, 0])  # Use sepal length as X
petal_length = np.array(iris.data[:, 2])   # Use petal length as Y

mbill = np.mat(sepal_length)
mtip = np.mat(petal_length)

m = np.shape(mbill)[1]
one = np.mat(np.ones(m))
X = np.hstack((one.T, mbill.T))

ypred = localWeightRegression(X, mtip, 0.5)
GraphPlot(X, ypred)
'''