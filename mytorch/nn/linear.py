import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features,1))

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A
        self.N = self.A.shape[0]  # TODO store the batch size of input
        self.ones=np.ones((self.N,1))
        # Think how will self.Ones helps in the calculations and uncomment below
        # self.Ones = np.ones((self.N,1))
        Z = np.matmul(A, np.transpose(self.W))+np.matmul(self.ones,np.transpose(self.b))

        return Z

    def backward(self, dLdZ):

        dZdA = np.transpose(self.W)
        dZdW = self.A
        dZdb = self.ones

        dLdA = np.matmul(dLdZ,np.transpose(dZdA))
        dLdW = np.matmul(np.transpose(dLdZ),dZdW)
        dLdb = np.matmul(np.transpose(dLdZ),dZdb)
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:

            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdb = dZdb
            self.dLdA = dLdA

        return dLdA
