import numpy as np
from pandas import crosstab

class ode:
    def __init__(self, laplace_estimate = 1, min_occurrence = 30):
        self.father = []
        self.Z = [] # Number of features
        self.Lap = laplace_estimate
        self.M = min_occurrence
        self.py = list()
        self.pxy = list()
        self.pxyx = list()
        self.names = dict()
        self.kind = 'empty'

    def fit_nb(self, X, y):

        C = len( set(y)) 
        self.py = np.array([ list(y).count(i) for i in set( y )], float ) / X.shape[0]
        self.pxy = list()
        self.Z = X.shape[1]
        self.names = 
        for z in range(self.Z):
            tmp = crosstab( X[:,z], y )
            tmp = np.asarray( tmp + self.Lap ).astype(float) / ( ( np.asarray( tmp +self.Lap ) ).sum(0))
            self.pxy.append( tmp )
            self.names[x] = map(int,list( set( X[:,z] )))
        self.kind = 'Naive Bayes'

    def fit_ode(self,X, y, father):
        # Calling the wrong estimator
        self.father = father

        self.Z = X.shape[1]
        C = len( set(y) )
        self.pxy = list()
        self.names = list()
        self.py = np.array([ list(y).count(i) for i in set( y )], float ) / X.shape[0]
        # P(x|y,x_father)

        for z in range(self.Z):
            if z is father:
                tmp = crosstab( X[:,self.father], y )
                tmp = np.asarray( tmp + self.Lap ).astype(float) / ( ( np.asarray( tmp +self.Lap ) ).sum(0))
                self.pxy.append( tmp )
                self.names[self.father] = map(int,list( set( X[:,self.father] ))) 
            else:
                for curr_y in set( self.y ):
                    tmp = crosstab( X[self.y == curr_y,z], X[self.y == curr_y,self.father] )
                    tmp = np.asarray( tmp + self.Lap ).astype(float) / ( ( np.asarray( tmp +self.Lap ) ).sum(0))
                    self.pxyx.append( tmp )
                    self.names[z] = map(int,list( set( X[self.y == curr_y,z] )))
        self.kind = 'One-Dependency Estimator'

class aode:

    def __init__(self):
        pass

    def fit_aode(self, X,y):
        self.Z = X.shape[1]
        for z in range( self.Z):
            pass
            
    def evaluate(self, X):
        pass

