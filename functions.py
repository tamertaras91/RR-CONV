import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Function to drive the graident w.r.t input of the convoutional layer
def drive_gradient(input_shape,weights,output_gradients,stride,padding):
    weights = torch.tensor(weights, requires_grad=True)
    input_tensor = torch.randn(input_shape, requires_grad=True)
    dL_dY = output_gradients
    # dummy forward pass to build the computational graph
    output = F.conv2d(input_tensor, weights, stride=stride, padding=padding)
    output.backward(dL_dY)
    dL_dX= input_tensor.grad
    return dL_dX
# Function to construct the input from convolutional layer using Gradients constraints

def construt_input_using_gradients(num_f,num_c,dim_x,output_gradient,weight_gradeint,padding,stride,kernal=5):
    input_matrix=dim_x*dim_x
    pad_dim=dim_x+2*padding
    a=np.array(output_gradient)
    Filters_gradients=np.array(weight_gradeint).reshape(num_f,num_c,kernal,kernal)
    x=[]
    indices=[]
    for n in range(num_c):
        cord_a=[]
        cord_b=[]
        rank=0
        for i in range(num_f):
            for k in range(kernal):
                for l in range(kernal):
                    if(rank==input_matrix):
                        break
                    cord_b.append(Filters_gradients[i][n][k][l])
                    array_gradients=np.zeros(pad_dim**2).reshape(pad_dim,pad_dim)
                    array_gradients[k:k+dim_x:stride,l:l+dim_x:stride]=a[i:i+1]
                    cord_a.append(array_gradients[padding:padding+dim_x,padding:padding+dim_x].reshape(input_matrix))
                    if(n==0):
                        current_rank=np.linalg.matrix_rank(cord_a)
                        if (current_rank==rank):
                            indices.append(i*kernal**2+k*kernal+l)
                            cord_a=cord_a[:-1]
                            cord_b=cord_b[:-1]
                        rank=current_rank
        if n!=0:
            cord_a=np.delete(cord_a,indices,axis=0)
            cord_b=np.delete(cord_b,indices,axis=0)
            cord_a=cord_a[0:input_matrix]
            cord_b=cord_b[0:input_matrix]
        sol=np.linalg.solve(cord_a,cord_b)
        sol2=sol.reshape(dim_x,dim_x)
        x.append(sol2)
    x=np.array(x).reshape(num_c,dim_x,dim_x)
    return x

# Function to construct the input from convolutional layer using weights constraints

def construt_input_using_weights(num_filters,num_c,dim_x,Y,W,pad,stride,kernal=5):
    a=[]
    b=[]
    dim=dim_x**2
    pdim=dim_x+pad
    for n in range(num_filters):
        q=0
        for k in range(0,dim_x,stride):
            v=0
            for l in range(0,dim_x,stride):
                a_row=np.zeros(dim_x**2*num_c)
                for c in range(num_c):
                    x1_=np.zeros((dim_x+2*pad)**2).reshape(dim_x+2*pad,dim_x+2*pad)
                    x1_[k:k+kernal,l:l+kernal]=W[n][c]
                    a_row[c*dim:dim+c*dim]=x1_[pad:pdim,pad:pdim].reshape(dim)
                a.append(a_row)
                b.append(Y[n][q][v])
                v+=1
            q+=1
    sol=np.linalg.solve(a[:dim_x**2*num_c],b[:dim_x**2*num_c]).reshape(num_c,dim_x,dim_x)
    return sol
