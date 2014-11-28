from csv import reader
import os
import numpy as np
from collections import Counter

def data_reader( name ):
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

    names_Z = list()
    features = np.zeros([T,Z])
    for z in range( Z ):
        listvalues = [row[z] for row in data]
        if len( set( listvalues ) ) > 10:
            names_Z.append('numeric')
            features[:,z] = [float(w) for w in listvalues]
        else:
            names_Z.append( list(set(listvalues)))
            features[:,z] = [list(set(listvalues)).index( w ) for w in listvalues]

    os.chdir(base_directory)
    return features, names_Z


def numerical2category( listvalues, B ):
    vals = [float( w ) for w in listvalues ]
    


