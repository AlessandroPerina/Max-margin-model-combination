import numpy as np
from pandas import crosstab

class aode:
    def __init__(self, laplace_estimate = 1, min_occurrence = 30):
        self.Fath = feature_father  
        self.Z = [] # Number of features
        self.Lap = laplace_estimate
        self.MinOc = min_occurrence
        if not self.Fath:
            self.type = 'ode'
        else:
            self.type = 'nb'
        self.pxy = []
        self.names = []

    def fit_nb(self, X, y):
        # Calling the wrong estimator
        if self.type == 'ode':
            self.fit_ode(X, y)
        Z = X.shape[1]
        C = len( set(y)) 
        self.py = np.array([ list(y).count(i) for i in set( y )], float ) / X.shape[0]
        self.pxy = list()
        self.Z = Z
        self.names = list()
        for z in range(self.Z):
            tmp = crosstab( X[:,z], y )
            tmp = np.asarray( tmp + self.Lap ).astype(float) / ( ( np.asarray( tmp +self.Lap ) ).sum(0))
            self.pxy.append( tmp )
            self.names = map(int,list( set( X[:,z] )))

    def fit_ode(self,X, y):
        # Calling the wrong estimator
        if self.type == 'nb':
            self.fit_nb(X, y)

        Z = X.shape[1]
        C = len( set(y))
        pxy = list()
        self.Z = Z
        self.names = list()
        for z in range(self.Z):
            tmp = crosstab( X[:,z], y )
            tmp = np.asarray( tmp + self.Lap ).astype(float) / ( ( np.asarray( tmp +self.Lap ) ).sum(0))
            self.pxy.append( tmp )
            self.names = map(int,list( set( X[:,z] )))   



    def evaluate(self, X):
        pass

