from sklearn.linear_model import LogisticRegression # importing Sklearn's logistic regression's module

# Essential libraries
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn utilities
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

'''
This part is for question 3
'''
# x_data = [[0.49,0.09],[1.69 ],[],[],[],[],[],[]]


'''
This func finds the max likelihood func
pass in sigmoid result h (computed from augmented raw input x) and actual label y
all three are supposed to be matrices
'''

h_test = [[0.389],[0.042],[0.613],[0.167],[0.572],[0.526],[0.393],[0.638]]
h_mat = np.array(h_test)

y = np.array([[0],[0],[0],[0],[1],[1],[1],[1]])

x = np.array([[0.49,0.09],[1.69,0.04],[0.04,0.64],[1,0.16],[0.16,0.09],[0.25,0],[0.49,0],[0.04,0.01]]) #raw data
x = np.hstack((np.ones((x.shape[0],1)),x))   #augmentation

# w = np.array([1.33,-2.96,-2.77])
w = np.array([0.66,-2.24,-0.18])

'''
Sigmoid function
w is 1 by n coeff matrix, where n is num of features
x is N by n data matrix after augmentation. N is sample size per feature

return sigmoid matrix, which is the h matrix we use later.
'''
def sigmoid(w,x):
    return np.reciprocal(np.ones((x.shape[0],1))+((np.exp((-w@x.T)).T).reshape(-1,1)))


'''
Max Likelihood Function

has cross entropy applied.
'''
def mlf(h,y):   #return just the value

    return -(y.T @ np.log(h) + ((np.ones((y.shape[0],1)) - y).T @ np.log(np.ones((h.shape[0],1))-h)))[0][0]
# print(np.exp(h_mat))
# print(y.shape)
# print(mlf(sigmoid(w,x), y))
# print(mlf(h_mat, y))

# print(sigmoid(w,x))
# print((np.ones((x.shape[0],1))+((np.exp((-w@x.T)).T).reshape(-1,1))).shape)

'''
Gradient ascent on entire w matrix
pass in orginial coeff matrix w,
x val --training sample
y val --labels for training sample
learning rate a,
iteration number iter
return improved coeff matrix
'''
def gradientAscent(w,x,y,a,iter):
    for i in range(iter):
        w = w + ((a/x.shape[0])*(x.T@(y-sigmoid(w,x)))).T
    return w


print(gradientAscent(w,x,y,0.1,1))
a = 0.1

# print(y.shape)
# mat1 = np.array([[1],[2],[3]])
# mat2 = np.array([[1,2],[2,4],[3,6]])
print((a/x.shape[0]))