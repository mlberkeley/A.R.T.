import numpy as np
from utils import *

Rfront=None
Rside=None
T=np.array([0,0,0])
def preprocess_all(filelist,PCA_fit=None,project_bool=False):
	for name in filelist:
		yield preprocess_single(name,PCA_fit=PCA_fit,project_bool=project_bool)

def preprocess_single(filename,points=None,PCA_fit=None,project_bool=False):
	if points is None:
		_1,points,_2=file_manop.parse_file(filename)
		del _1,_2
	if PCA_fit:
		return get_PCA_weights(points,PCA_fit)
	elif project_bool:
		front=get_projection(points,Rfront,T)
		side=get_projection(points,Rside,T)
		return front,side
	else:
		return points.flatten()

def get_PCA_weights(points,PCA_fit):
	return PCA_fit.transform(points.flatten())

def get_projection(points,R,T_):
	return (points.dot(R)+T_)[:,:-1]

