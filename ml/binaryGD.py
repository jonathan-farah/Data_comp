# TODO Q07
def gradient_descent(X_1 , y , learning_rate , w , n , num_iters):
    # In place of ellipsis, write the updated value of w0 in temp0 and of w1 in temp1
    # Finish the gradient descent function
    for i in range(num_iters):
        # derivative vector is given by : X_train.Transpose *  (( X_train * w)- y )
        # You may add your own variables
        w = w - (learning_rate)*(1/n)*X_1.T@(X_1@w-y) # derivative vec formula is wrong! missing 1/n
        # w = w0 - (learning_rate)* X_1.T @  (( X_1 @ w0)- y)
        # print(w[1])

    return w