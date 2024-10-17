##with hidden##


import numpy as np
def activfun_sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
x=np.array([[1,1,0,1],[0,0,0,0],[0,1,1,1],[1,1,1,1],[0,1,0,1],[0,1,0,0],[1,0,0,1],[0,1,0,0]])
y=np.array([[0,0,1,0,1,0,1,0]]).T
np.random.seed(1)
w0=2*np.random.random((4,4))-1
w1=2*np.random.random((4,1))-1
for i in range(140):
    L_input=x
    L1=activfun_sigmoid(L_input.dot(w0))
    L_output=activfun_sigmoid(L1.dot(w1))
    
    L_output_error=y-L_output
    
    L_output_delta=L_output_error*activfun_sigmoid(L_output,True)
    
    L1_error=L_output_delta.dot(w1.T)
    
    L1_delta=L1_error*activfun_sigmoid(L1,deriv=True)
    w1=w1+2*L1.T.dot(L_output_delta)
    w0=w0+2*L_input.T.dot(L1_delta)
print("Output after training")
print(L_output)
print("Loss:\n"+str(np.mean(np.square(y-L_output))))

##without hidden##

import numpy as np
def activfun_sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
x=np.array([[1,1,0,1],[0,0,0,0],[0,1,1,1],[1,1,1,1],[0,1,0,1],[0,1,0,0],[1,0,0,1],[0,1,0,0]])
y=np.array([[0,0,1,0,1,0,1,0]]).T
np.random.seed(1)
w0=2*np.random.random((4,4))-1
print("Initail weight-\n",w0)
for i in range(1000):
    L_input=x
    
    L_output=activfun_sigmoid(L_input.dot(w0))
    
    L_output_error=y-L_output
    
    L_output_delta=L_output_error*activfun_sigmoid(L_output,True)
    
   
  
    
    w0=w0+2*L_input.T.dot(L_output_delta)
print("Final weights=\n",w0)
print("Output after training")
print(L_output)
print("Loss:\n"+str(np.mean(np.square(y-L_output))))