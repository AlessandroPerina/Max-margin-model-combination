from csv import reader
import os
import numpy as np

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
		if (len(self.data[-1]) == 0):
			del self.data[-1]

		self.T = len( self.data )
		self.Z = len( self.data[0] )
		self.no_Z = [0 for i in range(0,self.Z )]
		self.values_Z = [0 for i in range(0,self.Z )]
		for z in range(0,self.Z ):
			self.values_Z[z] = np.unique( [row[z] for row in self.data]) 
			self.no_Z[z] = len(self.values_Z[z] )

		self.X = np.zeros([Z,T])
		


