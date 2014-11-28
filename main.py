#main
import os
import data_reader as DR

name = 'nursery'
os.chdir('C:\Users\APerina\Desktop\Git\max-margin-model-combination')
features, names =  DR.data_reader( name )

name = 'glass'
features2, names2 =  DR.data_reader( name )

print "Ho finito"

