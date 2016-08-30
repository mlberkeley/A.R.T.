import os
import art_preprocessing.process
from sklearn.decomposition import PCA
from sklearn import linear_model
import pickle as pkl
import numpy as np
TRAIN_DIR=''
PCA_PATH=''
LIN_MAP_PATH=''
sampling_criteria=[3,2]
def train_all(n_comps=None,front_comp=None,side_comp=None):
	trainlist=list(os.listdir(TRAIN_DIR))
	points_mat=np.array(list(art_preprocessing.process.preprocess_all(trainlist)))
	points_mat=sample_points(points_mat)
	if n_comps:
		PCA_fit=PCA(n_components=n_comps)
	else:
		PCA_fit=PCA()
	PCA_fit.fit(points_mat)
	del points_mat
	"""saving trained PCA module"""
	pkl.dump(PCA_fit,open(PCA_PATH,'wb'))

	weight_mat=np.array(list(art_preprocessing.process.preprocess_all(trainlist,PCA_fit=PCA_fit)))
	model=linear_model.LinearRegression()
	measurements=[]
	if front_comp:
		front_PCA=PCA(n_components=front_comp)
	else:
		front_PCA=PCA()
	if side_comp:
		side_PCA=PCA(n_components=side_comp)
	else:
		side_PCA=PCA()
	for front,side in art_preprocessing.process.preprocess_all(trainlist,project_bool=True):
		_,S1,V1=front_PCA._fit(front)
		_,S2,V2=side_PCA._fit(side)
		del _
		measurements.append(np.concatenate([S1.flatten(),V1.flatten(),S2.flatten(),V2.flatten()]))
	del side_PCA
	del front_PCA
	measurements=np.array(measurements)
	clf = linear_model.Lasso(alpha=0.1)
	clf.fit(measurements,weight_mat)
	pkl.dump(clf,open(LIN_MAP_PATH,'wb'))

def sample_points(points):
	return points[::sampling_criteria[0],::sampling_criteria[1]]