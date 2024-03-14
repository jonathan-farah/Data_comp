import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from polyRegress import poly_regression

'''
Global parameters:
N: Number of samples of each feature (how many rows, not counting the header)
d: Number of features (how many columns, not counting your sentiment result)
k: Number of folds of validation. This is slightly more advanced concept (only very teeny-tiny advanced)

I might add more in the future
'''
global N, d, k
N = 10000  #in the case of my test data, I know N is 33. It is also X.shape[0]
d = 1   #as above, also X.shape[1]
k = 3   #3 makes sense as 33 is divisible by 3, this is a designer choice

df = pd.read_csv("data2.csv")  # note that I assume the grader has the csv file of exactly this name and in same directory as the code.
data = df.values
X = data[:,0] # note that this is not column vec, need X.reshape(-1,1) to make it column vec
y = data[:,1]

# Next, we set nu_samp to be the number of examples, and N to be the 1/3 the number of original examples.
N //=k

# Reshape X and Y to be rank 2 matrices X_2d(<np.ndarray>) and y_2d(<np.ndarray>)
X = X.reshape(-1,1)
# In case we need augmentation, X_aug = np.hstack((np.ones((X.shape[0], 1)), X))

y = y.reshape(-1,1)
# y_aug = np.hstack((np.ones((y.shape[0],1)),y))

X_tr = X[:2*N]  #use first 2/3 of all data to train
X_val = X[2*N:] #use last 1/3 to validate

y_tr = y[:2*N]  #correspondingly y is divided to training set and validation set
y_val = y[2*N:]


'''
This block of code is trying to gradually overfit data
'''
lambda1 =0  # We set lambda1 = 0 since we are not using any regularization in part 2.

# Record (store) the training error, validation error and w in train_costs(<list>),
# validation_costs(<list>) and w_dict(<dict>) for degrees = 1, 2, 3,..., 10
validation_costs_6 = []
train_costs_6 = []
w_dict_6 = {}
print("first thing to check: ",X_tr.shape)

model_degree = range(1,11) # The different feature transformatons we will perform

for d in model_degree:
    print('Order: ', d)
    poly = PolynomialFeatures(d)  #initialize feature matrix, to be loaded with x training/validation
    X_tr_poly = poly.fit_transform(X_tr) # transforms the training data, dimension (11, 2->11)
    # print("X_val shape: ",X_val.shape)
    X_val_poly = poly.transform(X_val) # transforms the validation data, dimensions same as above
    # print("checking dimensions: xpoly: ",X_tr_poly.shape)

    w = poly_regression(X_tr_poly, y_tr, lambda1) # w is of dimension (2->11, 1), each a coefficient (in []) of each degree term
    w_dict_6[d] = w  # save the value of w

    yhat = X_tr_poly@w  #this is shape(11,1) for 11 predicted points

    E_in = (((yhat-y_tr).T@(yhat-y_tr))/N)[0][0]  #A.T@A is the square of A
    train_costs_6.append(E_in)
    # predict yhat_val for the validation data, compute E_val (MSE for the validation data) and store E_val in validation_costs
    yhat_val = X_val_poly@w
    E_val = (((yhat_val-y_val).T@(yhat_val-y_val))/N)[0][0]
    validation_costs_6.append(E_val)



    # print('w: ', w)
    # print("X_tr_poly size:", X_tr_poly.shape)
    # print("max:", np.max(X_tr_poly))
    # print('E_in: ',E_in)
    # print("E_val:", E_val)
    # print('-------------------------')
    # ################################################################
    # d = 10
    # print('Order: ', d)
    poly = PolynomialFeatures(d)
    X_tr_poly = poly.fit_transform(X_tr)  # transforms the training data
    X_val_poly = poly.fit_transform(X_val)  # transforms the validation data

    validation_costs_10 = []
    train_costs_10 = []
    w_dict_10 = {}

    lambda_values = np.logspace(-10, 1,10)  # After finding the best lambda value, we should go and try more lambda values near the best one we have tried.  However, our goal here to show how regularization works - so we will skip that step.
    # lambda_values = range(1,10)
    for lambda1 in lambda_values:
        # Use your function from part 1 to determine w.
        w = poly_regression(X_tr_poly, y_tr, lambda1)  # no changes
        w_dict_10[lambda1] = w  # save w for the current value of lambda1

        # predict yhat for the training data, compute E_in (MSE for the training data) and store E_in in train_costs
        yhat = X_tr_poly @ w  # no change
        E_in = (((yhat - y_tr).T @ (yhat - y_tr)) / N)[0][0]  # flatten to float level
        train_costs_10.append(E_in)

        # predict yhat_val for the validation data, compute E_val (MSE for the validation data) and store E_val in validation_costs
        yhat_val = X_val_poly @ w
        E_val = (((yhat_val - y_val).T @ (yhat_val - y_val)) / N)[0][0]
        validation_costs_10.append(E_val)


    poly = PolynomialFeatures(d)
    xp = np.linspace(-5, 5, 200).reshape((200, 1))
    xp_d = poly.fit_transform(xp)

    counter = 0
    for lambda1 in lambda_values:
        counter += 1
        plt.xlim([-5, 5])
        plt.ylim([-80, 85])
        # Plot data
        plt.plot(X_tr, y_tr, 'o')

        # Plot hypothesis as a line
        yp_hat = xp_d.dot(w_dict_10[lambda1])  # type the following: xp_d.dot(w_dict_10[lambda1])

        # Plot hypothesis
        # plt.plot(xp, yp_hat)
        # plt.xlim(-5, 5)
        #
        # plt.title('Training data & model of degree ' + str(d))
        # plt.xlabel('X')
        # plt.ylabel('y')
        # plt.grid(True)
        # plt.show(block=False)

    # print(type(train_costs_10))
print("best in validation set 6",np.argmin(validation_costs_6))

print("best in set 10",np.argmin(validation_costs_10))
# above parameter will be used as the best model selection
# we also need to trim off irrelevant features using lasso
# print(w_dict_10[6])
# yp_hat = xp_d.dot(w_dict_10[6])  # type the following: xp_d.dot(w_dict_10[lambda1])
# xp = np.linspace(-5, 5, 200).reshape((200, 1))
# plt.plot(xp, yp_hat)
# plt.xlim(-5, 5)
#
# plt.title('Training data & model of degree ' + str(d))
# plt.xlabel('X')
# plt.ylabel('y')
# plt.grid(True)
# plt.show(block=False)