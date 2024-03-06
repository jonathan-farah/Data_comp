import numpy as np


# Note, here we are using lambda1 instead of lambda because lambda is a reserved word in Python.
def poly_regression(X,y,lambda1): # We are using X for the design matrix.  It might be that X is in Z-space
    d = X.shape[1]
    N = X.shape[0]
    I_N = np.identity(d,dtype = float)  #using d not N because addition with X.T@X which is d by d
    w = (np.linalg.inv(X.T@X+N*lambda1*I_N))@X.T@y  #very useful regualrization formula
    return w

# print((np.identity(33,dtype = float)))
# print((y_2d.T@y_2d).shape)

'''
code below ensures linear regression is correct
'''
# test_x = np.ones((5,1))
# test_y = np.ones((5,1))
# test_y /=2
# print(poly_regression(test_x,test_y,0))


'''
checked against slides. Good.
'''