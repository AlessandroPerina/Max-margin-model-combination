from csv import reader
import os
import numpy as np
from collections import Counter

class data_reader:

	def __init__(self, name):

		base_directory = os.path.abspath(".")
		self.name = name
		self.data = list()
		os.chdir( ".\Datasets\\" + name )
		with open( name + ".data", "rb" ) as f:
			tmp = reader(f)
			for idx,val in enumerate(tmp):
				self.data.append(val)

		os.chdir( base_directory )
		while (len(self.data[-1]) == 0):
			del self.data[-1]

		self.T = len( self.data )
		self.Z = len( self.data[0] )
		self.no_Z = [0 for i in range(0,self.Z )]
		self.values_Z = dict()
		self.values_Z_D = dict()
		self.theta = list()
		self.thetaD = list(list())
		for z in range(0,self.Z ):
			listvalues = [row[z] for row in self.data]
			for z2 in range(0,self.Z):
				if z2 != z:
					listvalues2 = [row[z2] for row in self.data]
					tmp_hist = np.array( np.asarray( Counter( zip(listvalues, listvalues2) ).values() )+ 1, dtype = float)
					self.thetaD[z][z2] = tmp_hist / sum( tmp_hist)
					self.values_Z_D.update({[z,z2]:c.keys()})

			c = Counter( listvalues )
			self.values_Z.update({z:c.keys()})
			tmp_hist = (np.array( np.asarray( c.values() )+1 ,dtype=float)  )/ sum( np.asarray( c.values() )+1 )
			self.theta.append(tmp_hist) 
			self.no_Z[z] = len(self.values_Z[z] )	



