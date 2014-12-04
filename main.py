#main
import os
import numpy as np
from sklearn import cross_validation
skf = cross_validation.StratifiedKFold(y, n_folds= no_folds)
os.chdir('C:\Users\APerina\Desktop\Git\max-margin-model-combination')
try:
    __import__('imp').find_module('DR')
    reload( DR )
except:
    import data_reader as DR

no_folds = 3

#name = 'nursery'
#features, names =  DR.data_reader( name,  )
#data_reader( name, label_id = 'last', type = 'C', has_index = True )

name = 'glass'
raw_data, names, X, y =  DR.data_reader( name, 'last', 'N', True )

skf = cross_validation.StratifiedKFold(y, n_folds= no_folds)
data = list()
tmp = np.zeros( X.shape) 
for train_index, test_index in skf:

    X_train = X[train_index,:]
    X_test = X[test_index,:]
    y_train = y[train_index]
    y_test = y[test_index]

    for z in [i for i in range( len( X_train[1]) ) if names[i] == 'numerical']:
        xsort = list(  np.sort( X_train[:,z] ) )
        idsort = np.argsort( X_train[:,z] )
        ysort = list( y_train[idsort] )
        bins = DR.MDL_discretize(xsort,ysort, set( ysort)  )
        bins.sort()

        tmp[:,z] = np.digitize(X[:,z],bins)
    data.append(tmp)

