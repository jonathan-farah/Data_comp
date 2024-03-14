import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from polyRegress import poly_regression


'''
data1 has 605 rows, need to remove middle column
data2 has 10000 rows.
'''

df = pd.read_csv("data1.csv")  # note that I assume the grader has the csv file of exactly this name and in same directory as the code.
data = df.values
X = data[:,:1] # note that this is not column vec, need X.reshape(-1,1) to make it column vec
# y = data[:,d]
print(X.shape)