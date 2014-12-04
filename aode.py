import numpy as np

class aode:

    def __init__(self, feature_father = [], laplace_smoothing = 2, min_counts = 5):
        self.Fath = feature_father  
        self.Nof = 0
        self.Lap = 2
        self.minc = min_counts
        if not sefl.Fath:
            self.type = 'ode'
        else:
            self.type = 'nb'

    def fit_nb(self, X):
        pass

    def fit_ode(self,X):
        pass
    
    def evaluate(self, X):
        pass

