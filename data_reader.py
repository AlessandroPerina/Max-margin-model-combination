from csv import reader
import os
import numpy as np
from collections import Counter

def data_reader( name, label_id = 'last', type = 'C', has_index = True ):
    # C -> Categorical, N-> Numerical feature
    # label_id -> index of the labels 'last' or 'first'
    # has_index -> the first 'feature' is the ordinal index of the sample. Get rid of it
    
    assert label_id == 'last' or label_id == 'first'
    assert isinstance( has_index, bool)
    assert  has_index and label_id == 'last'

    try:
        base_directory = os.path.abspath(".")
        data = list()
        os.chdir( ".\Datasets\\" + name )
        with open( name + ".data", "rb" ) as f:
            tmp = reader(f)
            for idx,val in enumerate(tmp):
                data.append(val)
        # Cancella le ultime linee vuote (se esistono)
        while (len(data[-1]) == 0):
            del data[-1]
    
        T = len( data )
        Z = len( data[0] )
        # Default. All the features are categorical
        if type == 'N':
            kind_Z = list(type)+['N']*(Z-len(type))
        else:
            kind_Z = list(type)+['C']*(Z-len(type))
   

        names_Z = list()
        features = np.zeros([T,Z])
        for z in range( Z ):
            listvalues = [row[z] for row in data]
            if kind_Z[z] == 'N':
                names_Z.append('numerical')
                features[:,z] = [float(w) for w in listvalues]
            else:
                # If it is not numercial, it is categorical
                names_Z.append( list(set(listvalues)))
                features[:,z] = [list(set(listvalues)).index( w ) for w in listvalues]

        if has_index:
             y = features[:,-1]
             C = len( set( y ))
             X = features[:,1:-1]
        else:
            if label_id == 'last':
                y = features[:,-1]
                X = features[:,0:-1]
            else:
                y = features[:,0]
                X = features[:,1:]

        os.chdir(base_directory)
        return features, names_Z, X, y
    except:
        print "Error somewhere!"
        os.chdir('C:\Users\APerina\Desktop\Git\max-margin-model-combination')


def MDLdiscretize( x, y, no_values = 10):
    pass


def _MDLfind_partition( x,y ):
    xsort = [a for (a,b) in sorted( zip(x,y) )]
    ysort = [b for (a,b) in sorted( zip(x,y) )]
    pass


def _MLDpartition_value( xsort, ysort, idpartition ):
    # Return the cut-off point

    T = len(xsort)
    countc = [ ysort.count(a) for a in set(ysort)]

    boundary_point = list()
    for p in range(1,T-1):
        if ysort[p] != ysort[p-1]:
            boundary_point.append(p)

    E = np.zeros( len( boundary_point ));
    for b in enumerate( boundary_point ):
        tmp = ysort[0:b[1]+1]
        pcs1 = [ float( tmp.count(a) ) for a in set(tmp)] / countc

        tmp = ysort[b[1]+1:]
        pcs2 = [ float( tmp.count(a) ) for a in set(tmp)] / countc

        E[b[0]] = -( ((float(len(ysort[0:b[1]+1])-1))/T) *sum( pcs1*np.log( 1e-30 + pcs1 ) ) + ((float(len(ysort[b[1]+1:])-1))/T) *sum( pcs2*np.log( 1e-30 + pcs2 ) ) )

    return xsort[b[E.argmin()]]