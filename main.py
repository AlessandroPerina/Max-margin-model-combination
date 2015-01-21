#main

print "importing modules..."
import os
import numpy as np
from sklearn import cross_validation
from matplotlib import  pyplot as pp
os.chdir('C:\Users\wilde_000\Desktop\git\max-margin-model-combination')
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

pp.imshow(X,aspect=float( X.shape[1])/float( X.shape[0] ) ,interpolation='none')


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


# This is code to use in a separate QT Console!
# NOW I HAVE THE DATA FOR A FOLD 
counter = 0
for train_index, test_index in skf:
    X_train = data[counter][train_index,:]
    X_test = data[counter][test_index,:]
    y_train = y[train_index]
    y_test = y[test_index]

    

    counter +=1

import aode
import pdb  #debugger - in the case I need it
#Package name : aode
#Classes: ode, aode - Naive bayes is a ode without father
nb = aode.ode()
nb.fit_nb( X_train, y_train )
LL, y_predict = nb.ode_likelihood( X_test )
accuracy_nb = float( sum( y_predict == y_test ))*100 / len( y_test )

accuracy_ode = np.zeros([X_train.shape[1]])
for z in range(Z):
    od = aode.ode()
    od.fit_ode( X_train, y_train, z )
    LL, y_predict = od.ode_likelihood( X_test )
    accuracy_ode[z] = float( sum( y_predict == y_test ))*100 / len( y_test )
    del od

pp.plot(range(Z),accuracy_nb)
pp.plot(range(Z),accuracy_ode)

Sh = LL
pp.imshow(Sh,aspect=float( Sh.shape[1])/float( Sh.shape[0] ) ,interpolation='none')



'''
Sh = nb.pxy[0]
pp.imshow(Sh,aspect=float( Sh.shape[1])/float( Sh.shape[0] ) ,interpolation='none')

Sh = od.pxy[0]
pp.imshow(Sh,aspect=float( Sh.shape[1])/float( Sh.shape[0] ) ,interpolation='none')
'''