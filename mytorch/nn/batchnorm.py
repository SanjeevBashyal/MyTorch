import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        
        self.Z = Z
        self.N = self.Z.shape[0]
        self.C = self.Z.shape[1]
        self.M = np.sum(Z,axis=0)/self.N
        self.V = np.sum(np.square(Z-self.M),axis=0)/self.N

        if eval == False:
            # training mode
            self.NZ = (Z-self.M)/np.sqrt(self.V+self.eps)
            self.BZ = self.BW * self.NZ + self.Bb

            self.running_M = self.alpha*self.running_M + (1-self.alpha)*self.M
            self.running_V = self.alpha*self.running_V + (1-self.alpha)*self.V
        else:
            # inference mode
            self.NZ = (Z-self.running_M)/np.sqrt(self.running_V+self.eps)
            self.BZ = self.BW * self.NZ + self.Bb

        return self.BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.sum(dLdBZ*self.NZ,axis=0)
        self.dLdBb = np.sum(dLdBZ,axis=0)

        dLdNZ = dLdBZ * self.BW
        dLdV = -(1/2)*np.sum(dLdNZ*(self.Z-self.M)*(self.V+self.eps)**(-3/2),axis=0)
        dNZdM = - (self.V+self.eps)**(-1/2)-(1/2)*(self.Z-self.M)*(self.V+self.eps)**(-3/2)*(-2/self.N)*np.sum((self.Z-self.M),axis=0)
        dLdM = np.sum(dLdNZ * dNZdM,axis=0)

        dLdZ = dLdNZ * (self.V+self.eps)**(-1/2)  +  dLdV * ((2/self.N)*(self.Z-self.M))  +  (1/self.N)*dLdM

        return dLdZ
