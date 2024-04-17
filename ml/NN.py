'''
Author: Ren
(I think my prof or TA might be checking this code lol)

This is a teaching/learning tool
Does not intend to perform neural network training or prediction
give weight matrices, bias, outcome, and raw input,
produce an anatomy of the inner working of neural networks

produce:
    each level prediction (before and after activation)
    delta values on each node
    differential of loss against single weight at any layer (this is created from activation and delta)
        the above shoud not be stored in node, but produced via a different function on spot
    backward and forward propagation

Approach: oop of nodes.
'''


import numpy as np
'''
tentatively, each node does not know its own location (row/col)
but I will store all nodes in 2d array, and they will
look for each other in the same array via index referencing
this also means essentially my neurons are zero-indexed
example:

    arr[i][j] = Node(weight, bias)

'''
class Node:
    def __init__(self, weight=0):
        self.weight = weight    #a list of weights going out
        self.z = None   #the z val and a vals only created after we run with raw input
        self.a = None
        self.delta = None   #this one is only created after known a, z, and actual y
    def __repr__(self):
        return "Node with weights: "+ str(self.weight)

'''
each bias node contains a list of bias numbers going out to the next layer
all bias nodes are stored in list (not nested) corres to the layers

the differential of lass regarding each bias is literally the delta at next
layer, so we do not have to store it
'''
class BiasLayer:
    def __init__(self, bias):   #in case we have complicated bias in the future
        self.bias = bias


'''
The default activation
'''
def sigmoid(z):
    return 1/(1+np.exp(-z))

'''
Careful: plug in entire sigmoid expression, or use the number produced by sigmoid
'''
def sigmoid_derivative(sig):
    return sig*(1-sig)


def relu(z):
    return max(0,z)

'''
we use differential = 0 at 0.
'''
def relu_derivative(z):
    if z > 0: return 1
    return 0

'''
A helper function to get a single node (on next layer) z value 
from one layer of nodes and one layer of bias
to a singe node on next layer
the targetNodeInd is simply a top to bottom index position
no need to indicate the layer index

note that this is not modification in place. we need assign
'''
def returnZ(NodeLayer, BiasLayer, targetNodeInd):
    zVal = BiasLayer.bias[targetNodeInd]
    for eachNode in NodeLayer:
        zVal += eachNode.a * eachNode.weight[targetNodeInd]
    return zVal

'''
NodeMat: a list of list of nodes
BiasMat:a list of biases for all layers

this function will modify all the Nodes in place (update their z and a vals)
'''
def forward_propagation(NodeMat, BiasMat):#check the general dimension
    if(len(BiasMat) != len(NodeMat)-1):
        raise Exception("General Dimension mismatch")

    for i in range(len(BiasMat)):   #check each layer, the bias amount should match next node layer
        if(len(BiasMat[i].bias) != len(NodeMat[i+1])):
            raise Exception("layer wise bias-node count mismatch at layer: ",i)


    for layerInd in range(len(NodeMat)-1): #from each layer, propagate to next layer
        for eachNodeInd in range(len(NodeMat[layerInd+1])): # each index position in the next layer
            NodeMat[layerInd+1][eachNodeInd].z = returnZ(NodeMat[layerInd], BiasMat[layerInd], eachNodeInd)
            # if(layerInd+1 == 1):    #second layer relu
            NodeMat[layerInd+1][eachNodeInd].a = relu(NodeMat[layerInd+1][eachNodeInd].z)    #get activation
            # else:
            # NodeMat[layerInd+1][eachNodeInd].a = sigmoid(NodeMat[layerInd+1][eachNodeInd].z)    #get activation

    '''
    The X values are straightsaway used as first layer of NodeMat
    Note we are still not fixing zero indexing issue.
        
    Note that this algorithm will not set activation for the very first layer, layer0
    we manually set first layer's activation to be equal to the z 
    
    with the above step, we are done
    '''


'''
Now this is the hardest part of entire algorithm
backward propagation

since our purpose is a learning tool
instead of automating everything
we must first obtain the yPred value using forvard propagation
then use backward propagation to find all the deltas
yTrue, yPred should pass in as a list
'''
def backward_propagation(NodeMat, yPred, yTrue):

    # loss = 0.5 * (yPred - yTrue)**2  this is the loss function we used in class

    for indLast in range(len(NodeMat[-1])): #need to get last layer done first, set their deltas one by one
        NodeMat[-1][indLast].delta = (yPred[indLast] - yTrue[indLast]) * sigmoid_derivative(yPred[indLast])   # remember, delta is defined as differential betwwen j and z

    for layerInd in range(len(NodeMat)-2, 0, -1): #from each layer, propagate to prev layer, exclusive of output layer, exclusive of first(ind[0]) layer
        for eachNodeInd in range(len(NodeMat[layerInd])): # now we update one node at a time, same layer range as above
            localDelta = 0
            for nextLayerInd in range(len(NodeMat[layerInd+1])):    #accumulate based on each node on the next layer
                # if(layerInd == 1):
                localDelta += NodeMat[layerInd+1][nextLayerInd].delta * NodeMat[layerInd][eachNodeInd].weight[nextLayerInd] * relu_derivative(NodeMat[layerInd][eachNodeInd].a)
                # else:
                # localDelta += NodeMat[layerInd+1][nextLayerInd].delta * NodeMat[layerInd][eachNodeInd].weight[nextLayerInd] * sigmoid_derivative(NodeMat[layerInd][eachNodeInd].a)
            NodeMat[layerInd][eachNodeInd].delta =localDelta


def getdJdW(NodeMat, wto, wfrom, wlayer):
    return NodeMat[wlayer+1][wto].delta * NodeMat[wlayer][wfrom].a



if __name__ == "__main__":
    NodeMat = [[None, None, None], [None, None, None], [None, None]]    #set all weights
    NodeMat[0][0] = Node([1,-4,1])
    NodeMat[0][1] = Node([0,-4,1])
    NodeMat[0][2] = Node([1,1,-5])
    NodeMat[1][0] = Node([1,1])
    NodeMat[1][1] = Node([-1,1])
    NodeMat[1][2] = Node([1,2])
    NodeMat[2][0] = Node()
    NodeMat[2][1] = Node()

    BiasMat = [None,None]   #set all biases
    BiasMat[0] = BiasLayer([-2,6,1])
    BiasMat[1] = BiasLayer([-6,2])

    NodeMat[0][0].a = 1 #set raw input is equivalent to setting first layer activations
    NodeMat[0][1].a = 2
    NodeMat[0][2].a = 3

    yTrue = [1,0]
    yPred = [0,4]




    #
    #
    forward_propagation(NodeMat, BiasMat)


    backward_propagation(NodeMat,yPred, yTrue)


    layer = 2
    for eachInd in range(len(NodeMat[layer])):
        print("now, delta of layer ",layer+1,": ",NodeMat[layer][eachInd].a)
    # for eachNode in NodeMat[2]: #check hidden layer 2, in the middle, the second layer
    #     print("zval at right final layer: ",eachNode.z)
    #
    # for eachNode in NodeMat[1]: #check hidden layer 2, in the middle, the second layer
    #     print("activation at middle layer: ",eachNode.a)


    # yPred = [NodeMat[2][0].a,NodeMat[2][1].a]
    # yTrue = [1,0]
    # backward_propagation(NodeMat,yPred, yTrue)
    # print("delta1 at layer3: ", NodeMat[2][0].delta)

    # print("then, dJ dw on layer2, from node 1 to node 1: ", getdJdW(NodeMat,0,0,1))
    #
    # print("then, we want delta1 of layer 2: ", NodeMat[1][0])
    #
    # print("then, dJ dw on layer1, from node 1 to node 1: ", getdJdW(NodeMat,0,0,0))
    #
    # print("then, check a11: ",NodeMat[0][0].a)

    '''
    above is params for problem 1
    '''


    '''
    below is qn3 params
    '''
    # #
    # NodeMat = [[None, None], [None, None], [None]]    #set all weights
    # NodeMat[0][0] = Node([1,3])
    # NodeMat[0][1] = Node([2,4])
    # NodeMat
    # NodeMat[1][0] = Node([1])
    # NodeMat[1][1] = Node([2])
    # NodeMat[2][0] = Node()
    #
    # BiasMat = [None,None]   #set all biases
    # BiasMat[0] = BiasLayer([1,-1])
    # BiasMat[1] = BiasLayer([1])
    #
    # NodeMat[0][0].a = 0 #set raw input is equivalent to setting first layer activations
    # NodeMat[0][1].a = 1
    #
    # yTrue = [0]
    #
    # # for eachLayer in NodeMat:
    # #     for eachNode in eachLayer:
    # #         print(eachNode,end=" ")
    # #     print()
    #
    # forward_propagation(NodeMat,BiasMat)
    #
    # yPred = [NodeMat[-1][0].a]
    #
    # backward_propagation(NodeMat,yPred,yTrue)
    #
    # layer = 0
    # for eachInd in range(len(NodeMat[layer])):
    #     print("now, delta of layer ",layer+1,": ",NodeMat[layer][eachInd].a)
    #
