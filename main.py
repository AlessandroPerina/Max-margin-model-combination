#main
import os
os.chdir('C:\Users\APerina\Desktop\Git\max-margin-model-combination')
try:
    __import__('imp').find_module('DR')
    reload( DR )
except:
    import data_reader as DR


#name = 'nursery'
#features, names =  DR.data_reader( name,  )
#data_reader( name, label_id = 'last', type = 'C', has_index = True )

name = 'glass'
features2, names2, X, y =  DR.data_reader( name, 'last', 'N', True )
x = X[:,0]
print "discretize"

xsort = [a for (a,b) in sorted( zip(x,y) )]
ysort = [b for (a,b) in sorted( zip(x,y) )]

cut_off_list = DR.MDL_discretize(xsort,ysort,list())



