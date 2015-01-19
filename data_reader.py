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
            kind_Z.pop(0)
            kind_Z.pop(-1)
        else:
            if label_id == 'last':
                y = features[:,-1]
                X = features[:,0:-1]
                kind_Z.pop(-1)
            else:
                y = features[:,0]
                X = features[:,1:]
                kind_Z.pop(0)

        os.chdir(base_directory)
        return features, names_Z, X, y
    except:
        print "Error somewhere!"
        os.chdir('C:\Users\APerina\Desktop\Git\max-margin-model-combination')


def MDL_discretize( xsort, ysort, classes ):
    T = len(xsort)
    k = len( classes)

    cut_offs_list = list()
    if len(xsort) <= 2:
        return []
    else:
        k1, k2, S1, S2, entS1, entS2, entS, Tz, cut_off_index = MLD_find_cut_off( xsort,ysort, classes )
        S = S1+S2
        gain = entS - (S1/S)*entS1 - (S2/S)*entS2
        delta = np.log2( 3**k - 2) - k*entS + k1*entS1 + k2*entS2
        accept_cut = gain > ( np.log2( S - 1) / S) +  ( delta / S )

        cut_offs_list += [Tz]
        if accept_cut:
            if len( set( ysort[0:cut_off_index+1] )) > 1 and not not MDL_compute_boundary( xsort[0:cut_off_index+1], ysort[0:cut_off_index+1] ):
                cut_offs_list += MDL_discretize( xsort[0:cut_off_index+1],ysort[0:cut_off_index+1], classes )
            if len( set( ysort[cut_off_index+1:] )) > 1 and not not MDL_compute_boundary( xsort[cut_off_index+1:], ysort[cut_off_index+1:] ):
                cut_offs_list += MDL_discretize( xsort[cut_off_index+1:],ysort[cut_off_index+1:], classes )
        else:
            return cut_offs_list
        return cut_offs_list

def MLD_find_cut_off( xsort, ysort, classes ):
    eps = np.spacing(1)
    # Return the cut-off point
    T = len(xsort)
    countc = np.asarray( [ ysort.count(a) for a in classes] )
    entS = -sum( ( countc.astype(float) / T )*np.log2( eps + countc.astype(float) / T) )

    boundary_point = MDL_compute_boundary( xsort, ysort )

    E = np.zeros( len( boundary_point ));
    for b in enumerate( boundary_point ):
        tmp = ysort[0:b[1]+1]
        pcs1 = np.asarray( [ float( tmp.count(a) ) for a in classes] ) 
        pcs1 /= (eps + sum( pcs1))

        tmp = ysort[b[1]+1:]
        pcs2 = np.asarray( [ float( tmp.count(a) ) for a in classes] )
        pcs2 /= (eps + sum( pcs2))

        E[b[0]] = -( ((float(len(ysort[0:b[1]+1])))/T) *sum( pcs1*np.log2( eps + pcs1 ) ) + ((float(len(ysort[b[1]+1:])))/T) *sum( pcs2*np.log2( eps + pcs2 ) ) )

    cut_off_index = boundary_point[E.argmin()]
    k1 = len( set( ysort[0:cut_off_index+1] ))
    k2 = len( set( ysort[cut_off_index+1:] ))

    tmp = ysort[0:cut_off_index+1]
    pcs1 = np.asarray( [ float( tmp.count(a) ) for a in classes] ) / (eps + countc)
    entS1 = -sum( pcs1*np.log2( eps + pcs1 ) )
    S1 = len(  ysort[0:cut_off_index+1] )

    tmp = ysort[cut_off_index+1:]
    pcs2 = np.asarray( [ float( tmp.count(a) ) for a in classes] ) / (eps + countc)
    entS2 = -sum( pcs2*np.log2( eps + pcs2 ) )
    S2 = len(  ysort[cut_off_index+1:] )

    return k1, k2, S1, S2, entS1, entS2, entS, xsort[cut_off_index], cut_off_index


def MDL_compute_boundary( xsort, ysort ):

    T = len(xsort)
    boundary_point = list()
    for p in range(0,T-1):
        if ysort[p] != ysort[p+1] and xsort[p] != xsort[p+1]:
            boundary_point.append(p)
    return boundary_point
