import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
 
        self.W = np.zeros((out_features, in_features),dtype='f')
        self.b = np.zeros((out_features,1),dtype='f')

        self.debug = debug

    def forward(self, A):

        self.A = A
        self.N = self.A.shape[0]  # store the batch size of input
        self.Ones=np.ones((self.N,1))
        Z = A @ self.W.T + self.Ones @ self.b.T

        return Z

    def backward(self, dLdZ):

        dZdA = self.W.T
        dZdW = self.A
        dZdb = self.Ones

        dLdA = dLdZ @ dZdA.T
        dLdW = dLdZ.T @ dZdW
        dLdb = dLdZ.T @ dZdb
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:

            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdb = dZdb
            self.dLdA = dLdA

        return dLdA
