import neuralfoil as nf 
from aerosandbox.geometry import Airfoil as Af 
from aerosandbox import numpy as np 

airfoil = Af("NACA0006")

n_alpha, n_re = 100, 100

alpha_min, alpha_max = -5, 5
re_min, re_max = 1e6, 2e6 

alphas = np.linspace(alpha_min, alpha_max, n_alpha)



