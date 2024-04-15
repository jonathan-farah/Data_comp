
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# use seaborn plotting defaults
import seaborn as sns; sns.set()



'''
just doing math
'''
import matplotlib

'''
coeff is a list containing all coeffs plus the bias term
coord is x point , in list
y is y val
'''
def funcMarg(coef,coord,y):
    res = 0
    for i in range(len(coord)):
        res += coef[i]*coord[i] #one by one
    res += coef[-1] # add bias
    res *= y
    return res

coef1 = [3,-3,-3]
coef2 = [300,-300,-300]

print(funcMarg(coef1,[1,1.5],-1))

def geoMarg(funcMarg,coef):
    mag = 0
    coef.pop()
    for each in coef:
        mag += each*each
    mag = mag**0.5
    return funcMarg/mag

# print(geoMarg(funcMarg(),coef1))
# print(coef1)

###################################handling plot#######################################

print("-----------------\n")
for i in range(10,-1,-1):
    print(i)

