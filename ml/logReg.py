from sklearn.linear_model import LogisticRegression # importing Sklearn's logistic regression's module

# Essential libraries
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn utilities
from sklearn.datasets import load_breast_cancer
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

def mlf(h,y):   #return just the value

    return -(y.T @ np.log(h) + ((np.ones((y.shape[0],1)) - y).T @ np.log(np.ones((h.shape[0],1))-h)))[0][0]
# print(np.exp(h_mat))
print(y.shape)
print(mlf(h_mat, y))