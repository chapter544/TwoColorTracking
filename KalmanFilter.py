import numpy as np
class KalmanFilter(object):
    def __init__(self, F, H, Q, R):
       self.F = F
       self.H = H
       self.Q = Q
       self.R = R


    def filter(self,xprev, Pprev, z, z_valid):
        #print xprev
        #print Pprev
        #print z
        xpred = np.dot(self.F, xprev)
        Ppred = np.dot(np.dot(self.F, Pprev),self.F.T) + self.Q * np.eye(self.F.shape[0])

        if not z_valid:
            return xpred, Ppred

        # update
        z_error = z - np.dot(self.H, xpred)
        S = np.dot(np.dot(self.H, Ppred), self.H.T) + self.R * np.eye(self.H.shape[0])
        K = np.dot( np.dot(Ppred, self.H.T), np.linalg.inv(S) )
    
        # filtering
        xhat = xpred + np.dot(K, z_error)
        Ppost = np.dot(np.eye(self.F.shape[0]) - np.dot(K, self.H), Ppred)
    
        return xhat, Ppost
