import numpy as np


class MSELoss:

    def forward(self, A, Y):

        self.A = A
        self.Y = Y
        self.N = self.A.shape[0]
        self.C = self.A.shape[1]

        Ones_N=np.ones((self.N,1))
        Ones_C=np.ones((self.C,1))

        se = np.square(A-Y)
        sse = Ones_N.T @ se @ Ones_C
        mse = sse/(2*self.N*self.C)

        return mse

    def backward(self):

        dLdA = (self.A - self.Y)/(self.N*self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):

        self.A = A
        self.Y = Y
        self.N = self.A.shape[0]
        self.C = self.A.shape[1]

        Ones_N=np.ones((self.N,1))
        Ones_C=np.ones((self.C,1))

        self.softmax = np.exp(A) / np.sum(np.exp(A),axis=1)[:,None]
        crossentropy = np.multiply(-Y,np.log(self.softmax)) @ Ones_C
        sum_crossentropy = Ones_N.T @ crossentropy
        L = sum_crossentropy / self.N

        return L

    def backward(self):

        dLdA = self.softmax-self.Y

        return dLdA
