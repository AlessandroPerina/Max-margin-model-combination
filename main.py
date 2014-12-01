#main
import os
os.chdir('C:\Users\APerina\Desktop\Git\max-margin-model-combination')
import data_reader as DR
name = 'nursery'
features, names =  DR.data_reader( name,  )
data_reader( name, label_id = 'last', type = 'C', has_index = True )

name = 'glass'
features2, names2, X, y =  DR.data_reader( name, 'last', 'N', True )

print "Ho finito"

