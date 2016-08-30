import pickle as pkl
import art_preprocessing.process
import art_train.train
import numpy as np
from utils import *
PCA_PATH=''
LIN_MAP_PATH=''
def predict_filelist(filelist,front_comp=None,side_comp=None,outlist=None):
	if front_comp:
		front_PCA=PCA(n_components=front_comp)
	else:
		front_PCA=PCA()
	if side_comp:
		side_PCA=PCA(n_components=side_comp)
	else:
		side_PCA=PCA()
	outbool=(outlist is not None)
	clf=pkl.load(open(LIN_MAP_PATH,'rb'))
	PCA_fit=pkl.load(open(PCA_PATH,'rb'))
	for i,f in enumerate(filelist):
		hdr,points,faces=file_manip.parse_file(f)
		# front,side=art_preprocessing.process.preprocess_single(points=points,project_bool=True)
		# _,S1,V1=front_PCA._fit(front)
		# _,S2,V2=side_PCA._fit(side)
		# del _
		# measurement=np.concatenate([S1.flatten(),V1.flatten(),S2.flatten(),V2.flatten()])
		# middle_prediction=clf.predict(measurement)
		# del measurement
		# out=PCA_fit.inverse_transform(middle_prediction)
		out=predict_from_points(points,clf=clf,front_PCA=front_PCA,side_PCA=side_PCA,PCA_fit=PCA_fit)
		rss= np.sum(np.square(points.flatten()-out))
		if outbool:
			file_manip.writeout(hdr,out.reshape(points.shape),faces,outlist[i])
		yield out,rss

def predict_from_points(points,clf=None,front_PCA=None,side_PCA=None,PCA_fit=None,comps=[]):
	if comps!=[]:
		if front_comp:
		front_PCA=PCA(n_components=comps[0])
		else:
			front_PCA=PCA()
		if side_comp:
			side_PCA=PCA(n_components=comps[1])
		else:
			side_PCA=PCA()
	if PCA_fit is None:
		PCA_fit=pkl.load(open(PCA_PATH,'rb'))
	if clf is None:
		clf=pkl.load(open(LIN_MAP_PATH,'rb'))
		
	front,side=art_preprocessing.process.preprocess_single(points=points,project_bool=True)
	_,S1,V1=front_PCA._fit(front)
	_,S2,V2=side_PCA._fit(side)
	del _
	measurement=np.concatenate([S1.flatten(),V1.flatten(),S2.flatten(),V2.flatten()])
	middle_prediction=clf.predict(measurement)
	del measurement
	return PCA_fit.inverse_transform(middle_prediction)
