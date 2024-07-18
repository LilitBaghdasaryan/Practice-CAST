import torch
import numpy as np
import torch.nn as nn


class NeuralNetwork():
    def __init__(self):
        self.W1 = torch.randn(7, 5)
        self.B1 = torch.randn(7, 1)

        self.W2 = torch.randn(4, 7)
        self.B2 = torch.randn(4, 1)

        self.relu = ReLU(X)

    def forward(self, X):
        X = X.view(5, -1)
        Z = torch.matmul(self.W1, X) + self.B1  #fc1   
        A = ReLU(Z)
        a1_z1 = [X, Z]

        Z = torch.matmul(self.W2, A) + self.B2  #fc2   
        Y_hat = ReLU(Z)
        a2_z2 = [A, Z]

        A_Z = [a1_z1, a2_z2]
        return Y_hat, A_Z

    def update_weights(self, dL_dW, dL_dB, learning_rate):
        self.W2 -= learning_rate * dL_dW[1]
        self.B2 -= learning_rate * dL_dB[1]
        self.W1 -= learning_rate * dL_dW[0]
        self.B1 -= learning_rate * dL_dB[0]

    def backward(self, num_of_layers):
        return backprop(num_of_layers)

    def get_W(self, i):
        if i == 1:
            return self.W1
        return self.W2

def RMSELoss(Y, Y_hat):
    res = ((Y - Y_hat) ** 2).mean().sqrt()
    return res

def RMSELoss_Yhat_derivative_mean(Y, Y_hat):
    n = Y_hat.size(1) 
    rmse = RMSELoss(Y, Y_hat)
    derivative = (2/n) * (Y_hat - Y) / rmse
    return derivative

def ReLU(X):
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j] = max(X[i][j], 0.)
    return X

def ReLU_Z_derivative(Z):
    deriv = []
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            ls = []
            if Z[i][j] == 0:
                ls.append(0.)
            else:
                ls.append(1.)
        deriv.append(ls)
    return torch.tensor(deriv)

def Z_W_derivative(X):
    return X

def Z_X_derivative(W):
    return W


X = torch.randn(20, 5)
Y = torch.rand(20, 1) * 10    

model = NeuralNetwork()

learning_rate = 0.03


def backprop(num_of_layers):
    for epoch in range(1000):
        loss_sum = 0
        grad_W_sum = [] 
        grad_B_sum = []
        for m in range(len(X)):
            Ym_hat, A_Z = model.forward(X[m])
            loss = RMSELoss(Y[m], Ym_hat)
            loss_sum += loss.item()

            dL_dA = RMSELoss_Yhat_derivative_mean(Y[m], Ym_hat) 

            dL_dX = dL_dA 

            for i in range(num_of_layers - 1, -1, -1):
                Ai = A_Z[i][0]
                Zi = A_Z[i][1]

                dA_dZ = ReLU_Z_derivative(Zi) 
                dZ_dW = Z_W_derivative(Ai) 
                dZ_dX = Z_X_derivative(model.get_W(i - 1)) 

                dL_dZ = dL_dX * dA_dZ 
                dL_dW = torch.matmul(dL_dZ, dZ_dW.T)
                dL_dB = dL_dZ
                
                if i != 0:
                    dL_dX = torch.matmul(dZ_dX.T, dL_dZ)

                if m == 0:
                    grad_W_sum.append(dL_dW)
                    grad_B_sum.append(dL_dB)
                    grad_W_sum.reverse()
                    grad_B_sum.reverse()
                else:
                    grad_W_sum[i] += dL_dW
                    grad_B_sum[i] += dL_dB

        avg_grad_W = [g / len(X) for g in grad_W_sum]
        avg_grad_B = [g / len(X) for g in grad_B_sum]
        print(loss_sum / len(X))
        model.update_weights(avg_grad_W, avg_grad_B, learning_rate)

model.backward(2)
