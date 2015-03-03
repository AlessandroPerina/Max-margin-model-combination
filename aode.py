# Statements to initalize the module. These are executed ONLY the first time
# the module name is encountered
import numpy as np
from pandas import crosstab


class ode:

    def __init__(self, min_occurrence = 10,  laplace_estimate = 1):
        self.father = []
        self.Z = [] # Number of features
        self.Lap = laplace_estimate # constant to add for laplace estimate
        self.M = min_occurrence # Min number of occurrences to collect robust statistics
        self.py = list() # List of classes
        self.pxy = list() # P(x = a | y) for each y
        self.pxyx = list() # P(x = a | y, x_father = b) for each y
        self.names = dict() # feature_nr : values_in_train_set
        self.kind = 'empty' # empty or Naive Bayes or One-Dependency Esitimator
        self.Zval = list()
        self.C = []
        self.M = min_occurrence
        self.validity = list()

    def fit_nb(self, X, y):
        self.C = map(int, list( set(y)) )
        self.py = np.array([ list(y).count(i) for i in set( y )], float ) / X.shape[0]
        self.Z = X.shape[1]
        self.Zval = map( len, map( np.unique, X.T ) )

        for z in range(self.Z):
            self.names[z] = map(int,list( set( X[:,z] )))
            ct = crosstab( X[:,z], y )
            ct = ct.reindex_axis( self.names[z], axis=0).fillna(0)
            ct = ct.reindex_axis( self.C, axis=1).fillna(0)
            tmp = np.asarray ( (ct + self.Lap).apply(lambda r: r/r.sum(), axis=0) )
            tmp = tmp.T
            self.pxy.append( tmp ) # Trasposition for a better indexing
            self.pxyx.append( None )
            self.names[z] = map(int,list( set( X[:,z] )))
        self.kind = 'Naive Bayes'



    def fit_ode(self, X, y, father):
        self.father = father
        self.Z = X.shape[1] # No of features
        self.Zval = map( len, map( np.unique, X.T ) )
        self.C = map(int, list( set(y)) )
        self.py = np.array([ list(y).count(i) for i in set( y )], float ) / X.shape[0]
        self.names[self.father] = map(int,list( set( X[:,self.father] )))
        self.validity = map( int, np.histogram( X[:,self.father], np.unique(X[:,self.father] ))[0] > self.M) 

        for z in range(self.Z):
            self.names[z] = map(int,list( set( X[:,z] )))
            if z is father:
                self.names[z] = map(int,list( set( X[:,z] )))
                ct = crosstab( X[:,z], y )
                ct = ct.reindex_axis( self.names[z], axis=0).fillna(0)
                ct = ct.reindex_axis( self.C, axis=1).fillna(0)
                tmp = np.asarray ( (ct + self.Lap).apply(lambda r: r/r.sum(), axis=0) )
                tmp = tmp.T
                self.pxy.append( tmp )
                self.pxyx.append( None )
            else:
                tmp_array = list() 
                for curr_y in set( y ):
                    ct = crosstab( X[y == curr_y,z], X[y == curr_y,self.father] )
                    ct = ct.reindex_axis( self.names[z], axis=0).fillna(0)
                    ct = ct.reindex_axis( self.names[self.father], axis=1).fillna(0)
                    pxx = np.asarray ( (ct + self.Lap).apply(lambda r: r/r.sum(), axis=0) )
                    tmp_array.append( pxx.T ) # Trasposition for a better indexing
                self.pxyx.append( tmp_array )
                self.pxy.append( None )

        self.kind = 'One-Dependency Estimator'

    def ode_likelihood(self, X):
        # 
        [T,Z] = X.shape
        LL = np.zeros([T,len(self.C)])
        id_has_father = [i for i in range(self.Z) if self.pxyx[i] is not None]
        id_no_father = [i for i in range(self.Z) if self.pxy[i] is not None] 
        for y_cur in range( len(self.C)):
            for t in range(T):
                LL[t,y_cur] = np.log( self.py[y_cur] ) + \
                sum( [np.log( self.pxy[z][y_cur][self.names[z].index( X[t][z] )] ) for z in id_no_father if X[t][z] in self.names[z] ] ) + \
                sum( [self.validity[int( X[t][self.father] )-1]*np.log( self.pxyx[z][y_cur][self.names[self.father].index( X[t][self.father] )][self.names[z].index( X[t][z] )] ) for z in id_has_father if X[t][z] in self.names[z] and X[t][self.father] in self.names[self.father] ] )

        y_predict = np.asarray( [self.C[i] for i in np.argmax(LL,axis=1)] )
        return LL, y_predict


    '''
                for z in id_no_father:
                    idz = int(X[t][z])-1
                    LL[t,y_cur] += np.log( self.pxy[z][y_cur][idz] 
                for z in id_has_father:
                    idz = int(X[t][z])-1
                    idf = int(X[t][self.father])-1
                    LL[t,y_cur] += np.log( self.pxyx[z][y_cur][idf][idz] )
   '''     

'''
def generate_matrices( odeobj ):
    try:
        M = max( odeobj.Zval )
        C = len( odeobj.C )
        Mpxy = np.zeros([odeobj.Z,C,M]).astype(float)
        Mpxyx = np.zeros([odeobj.Z,C,M,M]).astype(float)
        for z in range(Z):
            for y in odeboj.C:



    except Exception, e:
        raise e
'''


'''
class aode:

    def __init__(self):
        pass

    def fit_aode(self, X,y):
        self.Z = X.shape[1]
        for z in range( self.Z):
            pass
    
    def evaluate(self, X):
        pass
   '''     
