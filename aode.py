import numpy as np
from pandas import crosstab

class aode:
    def __init__(self, laplace_estimate = 1, min_occurrence = 30):
        self.Fath = []
        self.Z = [] # Number of features
        self.Lap = laplace_estimate
        self.M = min_occurrence
        self.pxy = []
        self.names = []

    def fit_nb(self, X, y):

        C = len( set(y)) 
        self.py = np.array([ list(y).count(i) for i in set( y )], float ) / X.shape[0]
        self.pxy = list()
        self.Z = X.shape[1]
        self.names = list()
        for z in range(self.Z):
            tmp = crosstab( X[:,z], y )
            tmp = np.asarray( tmp + self.Lap ).astype(float) / ( ( np.asarray( tmp +self.Lap ) ).sum(0))
            self.pxy.append( tmp )
            self.names = map(int,list( set( X[:,z] )))

    def fit_ode(self,X, y, father):
        # Calling the wrong estimator
        self.Fath = father

        self.Z = X.shape[1]
        C = len( set(y))
        pxy = list()
        self.names = list()
        for z in range(self.Z):
            tmp = crosstab( X[:,z], y )
            tmp = np.asarray( tmp + self.Lap ).astype(float) / ( ( np.asarray( tmp +self.Lap ) ).sum(0))
            self.pxy.append( tmp )
            self.names = map(int,list( set( X[:,z] )))   


    def fit_aode(self, X,y):
        self.Z = X.shape[1]
        for z in range( self.Z):
            pass
            
    def evaluate(self, X):
        pass

