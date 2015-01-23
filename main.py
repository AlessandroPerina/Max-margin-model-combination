#main

print "importing modules..."
import imp
import os
import numpy as np
from sklearn import cross_validation
from matplotlib import  pyplot as pp
import scipy.io as sio
os.chdir('C:\Users\APerina\Desktop\Git\max-margin-model-combination')
try:
    __import__('imp').find_module('DR')
    reload( DR )
except:
    import data_reader as DR


no_folds = 10

#name = 'nursery'
#raw_data, names, X, y =  DR.data_reader( name, 'last', 'C', True )

name = 'glass'
raw_data, names, X, y =  DR.data_reader( name, 'last', 'N', True )

C = set(y)
skf = cross_validation.StratifiedKFold(y, n_folds= no_folds)
data = list()

Z = X.shape[1]
Z_per_fold=np.zeros( [no_folds,Z] )
counter = 0
NoBins = np.zeros( [no_folds,Z] ) 
for train_index, test_index in skf:

    X_trainD = X[train_index,:]
    X_testD = X[test_index,:]
    y_trainD = y[train_index]
    y_testD = y[test_index]
    tmp = np.zeros( X.shape) 
    for z in [i for i in range( len( X_trainD[1]) ) if names[i] == 'numerical']:
        xsort = list(  np.sort( X_trainD[:,z] ) )
        idsort = np.argsort( X_trainD[:,z] )
        ysort = list( y_trainD[idsort] )
        bins = DR.MDL_discretize(xsort,ysort, set( ysort)  )
        bins.sort()

        Z_per_fold[counter,z] = len( bins ) + 1
        tmp[:,z] = np.digitize(X[:,z],bins,right=True )

    Xtmp = tmp

    data.append(tmp)
    counter +=1


# This is code to use in a separate QT Console!
# NOW I HAVE THE DATA FOR A FOLD 
import aode
import imp
try:
    imp.find_module('aode')
    reload( aode )
except ImportError:
    import aode

import pdb  #debugger - in the case I need it
#Package name : aode
#Classes: ode, aode - Naive bayes is a ode without father
accuracy_ode = np.zeros([no_folds,X.shape[1]])
accuracy_nb = np.zeros(no_folds)
accuracy_aode = np.zeros(no_folds)


counter = 0
for train_index, test_index in skf:

    X_train = data[counter][train_index,:]
    X_test = data[counter][test_index,:]
    y_train = y[train_index]
    y_test = y[test_index]
    T = X_test.shape[0]    
    LL_aode = np.zeros([T,Z])
    Px = np.zeros([len(C),Z,1,X_train.shape[0]])
    Px_te = np.zeros([len(C),Z,1,X_test.shape[0]])

    LL_aode = np.zeros([T,len( set( y_train) )])

    nb = aode.ode()
    nb.fit_nb( X_train, y_train )
    LLnb, y_predict = nb.ode_likelihood( X_test )
    accuracy_nb[counter] = float( sum( y_predict == y_test ))*100 / len( y_test )

    for z in range(Z):
        od = aode.ode(0)
        od.fit_ode( X_train, y_train, z )
        LL, y_predict = od.ode_likelihood( X_test )
        Px_te[:,z,0,:] = LL.T
        LL_aode += LL
        accuracy_ode[counter,z] = float( sum( y_predict == y_test ))*100 / len( y_test )
        LL, y_predict = od.ode_likelihood( X_train )
        Px[:,z,0,:] = LL.T
        del od

    sio.savemat(name + "_fold_" + str(counter+1), {'Px':Px, 'Px_te':Px_te, 'y':y_train, 'y_te':y_test})
    y_predict = np.asarray( [list( set( y_train))[i] for i in np.argmax(LL_aode,axis=1)] )
    accuracy_aode[counter] = float( sum( y_predict == y_test ))*100 / len( y_test )
    del nb
    counter +=1
    

#counter +=1
 #pp.plot(range(Z),np.tile(accuracy_nb.mean(),Z),'ys',range(Z),np.tile(accuracy_aode.mean(),Z), range(Z),accuracy_ode.mean(0),'rs')

# OTHER COMBINATION METHODS
accuracy_ode = np.zeros([no_folds,X.shape[1]])
accuracy_nb = np.zeros(no_folds)
accuracy_aode = np.zeros(no_folds)
accuracy_cll = np.zeros(no_folds) 
accuracy_mse = np.zeros(no_folds) 
accuracy_rbc = np.zeros(no_folds)
accuracy_allm = np.zeros(no_folds)
import objfun as of
from scipy.optimize import fmin_l_bfgs_b
counter = 0
for train_index, test_index in skf:

    X_train = data[counter][train_index,:]
    X_test = data[counter][test_index,:]
    y_train = y[train_index]
    y_test = y[test_index]
    TE = X_test.shape[0]   
    T = X_train.shape[0]
         
    LL_aode = np.zeros([TE,Z])
    Px = np.zeros([len(C),Z,1,X_train.shape[0]])
    Px_te = np.zeros([len(C),Z,1,X_test.shape[0]])

    LL_aode = np.zeros([TE,len( set( y_train) )])

    nb = aode.ode()
    nb.fit_nb( X_train, y_train )
    LLnb, y_predict = nb.ode_likelihood( X_test )
    accuracy_nb[counter] = float( sum( y_predict == y_test ))*100 / len( y_test )

    for z in range(Z):
        od = aode.ode(0)
        od.fit_ode( X_train, y_train, z )
        LL, y_predict = od.ode_likelihood( X_test )
        Px_te[:,z,0,:] = LL.T
        LL_aode += LL
        accuracy_ode[counter,z] = float( sum( y_predict == y_test ))*100 / len( y_test )
        LL, y_predict = od.ode_likelihood( X_train )
        Px[:,z,0,:] = LL.T
        del od

    sio.savemat(name + "_fold_" + str(counter+1), {'Px':Px, 'Px_te':Px_te, 'y':y_train, 'y_te':y_test})
    y_predict = np.asarray( [list( set( y_train))[i] for i in np.argmax(LL_aode,axis=1)] )
    accuracy_aode[counter] = float( sum( y_predict == y_test ))*100 / len( y_test )
    del nb

    M = Px.shape[1]
    K = Px.shape[2]
    TE = Px_te.shape[-1]

    lPx_true = np.zeros((Z,T))
    lPx_all = np.reshape( Px, [Cl,Z,T],order='F')
    lPx_te_all = np.reshape( Px_te, [Cl,Z,TE],order='F')

    for t in range(T):
        lPx_true[:,t] = np.reshape( Px[list( set(y) ).index(y_train[t]),:,:,t], Z,order='F')
    
    w0 = np.ones(Z)
    bounds = np.tile((0,1),[Z,1])
    for z in range(Z): # ho un ensemble di Z classificatori
        w0[z]=np.random.random()

    #w0 = np.ones(Z)
    w0 /= sum(w0)
    (wf,f,d) = fmin_l_bfgs_b(of.cll, w0, fprime=None, args=(lPx_true, lPx_all), approx_grad=1 ,bounds=bounds)
    wf /= sum(wf)
    y_predict = [list( set(y) )[item] for item in list((lPx_te_all * wf[:None,None]).sum(1).argmax(0) )]
    accuracy_cll[counter] = float( sum( y_predict == y_test ))*100 / len( y_test )


    for z in range(Z): # ho un ensemble di Z classificatori
        w0[z]=np.random.random()
    py_true = np.zeros([Cl,T])
    for t in range(0,T):
        py_true[list(set(y)).index(y[t]),t] = 1
    (wf,f,d) = fmin_l_bfgs_b(of.mse, w0,fprime=None, args=(py_true, lPx_all), approx_grad=1 ,bounds=bounds)
    wf /= sum(wf)
    y_predict = [list( set(y) )[item] for item in list((lPx_te_all * wf[:None,None]).sum(1).argmax(0) )]
    accuracy_mse[counter] = float( sum( y_predict == y_test ))*100 / len( y_test )


    alpha0 = np.zeros(Z)
    for z in range(Z):
        alpha0[z]=np.random.random()
    (wf,f,d) = fmin_l_bfgs_b( of.rbc, alpha0,fprime=None, args=(lPx_true, lPx_all ), approx_grad=1 ,bounds=bounds)
    wf /= sum(wf)
    y_predict = [list( set(y) )[item] for item in list((lPx_te_all * wf[:None,None]).sum(1).argmax(0) )]
    accuracy_rbc[counter] = float( sum( y_predict == y_test ))*100 / len( y_test )
    

    lamb = 1
    wfc = np.zeros([Cl,Z])
    for c in range(Cl):
        for z in range(Z):
            w0[z]=np.random.random()

        (wf,f,d) = fmin_l_bfgs_b( of.allr, w0,fprime=None, args=(lPx_all, c, np.array( [list( C).index(z) for z in y_train] ), lamb ), approx_grad=1 ,bounds=bounds,maxiter=300)
        wfc[c,:] = wf    
            
    wfcn = wfc / wfc.sum(1)[:,None]
    y_predict = [list( set(y) )[item] for item in list((lPx_te_all * wfcn[:,:,None]).sum(1).argmax(0) )]
    accuracy_allm[counter] = float( sum( y_predict == y_test ))*100 / len( y_test )



    counter +=1

pp.bar(range(5), np.array([accuracy_nb.mean(),accuracy_aode.mean(),accuracy_cll.mean(),accuracy_mse.mean(),accuracy_rbc.mean()]))








'''
Sh = nb.pxy[0]
pp.imshow(Sh,aspect=float( Sh.shape[1])/float( Sh.shape[0] ) ,interpolation='none')

Sh = od.pxy[0]
pp.imshow(Sh,aspect=float( Sh.shape[1])/float( Sh.shape[0] ) ,interpolation='none')
'''