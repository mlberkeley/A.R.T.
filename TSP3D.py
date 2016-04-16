import numpy as np
clamp_value=1e-20
def TPS(fiducial_points,ctrlpoints,all_points):
	len_=len(fiducial_points)
	K=np.zeros((len_,len_))
	for r in range(len_):
		for c in range(len_):
			temp=fiducial_points[r]-fiducial_points[c]
			K[r,c]=temp.T.dot(temp)
			K[c,r]=K[r,c]
	K=np.maximum(K,clamp_value)
	K=np.sqrt(K)
	P=np.concatenate((np.ones((len_,1)),fiducial_points),axis=1)
	t1=np.concatenate((K,P),axis=1)
	t2=np.concatenate((P.T,np.zeros((4,4))),axis=1)
	L=np.concatenate((t1,t2))
	params=np.linalg.pinv(L).dot(np.concatenate((ctrlpoints,np.zeros((4,3)))))

	numpoints=len(all_points)
	K=np.zeros((numpoints,len_))
	all_x=all_points[:,0]
	all_y=all_points[:,1]
	all_z=all_points[:,2]
	

	for n in range(len_):
		K[:,n]=np.square(all_x-fiducial_points[n,0])+np.square(all_y-fiducial_points[n,1])+np.square(all_z-fiducial_points[n,2])
	K=np.maximum(K,clamp_value)
	K=np.sqrt(K)
	all_z=all_z.reshape((len(all_z),1))
	all_y=all_y.reshape((len(all_y),1))
	all_x=all_x.reshape((len(all_x),1))

	P=np.concatenate((np.ones((numpoints,1)),all_x,all_y,all_z),axis=1)
	L=np.concatenate((K,P),axis=1)
	rtn=L.dot(params)
	return rtn