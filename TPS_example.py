import numpy as np
from TPS3D import TPS

INPATH='Sculpt.off'
OUTPATH='Sculpt_TPS.off'


def extract_verts(path):
	headers=''
	verts=[]
	faces=''
	with open(path,'rb') as f:
		headers+=f.readline()
		temp=f.readline()
		num_vert=int(temp.split()[0])
		headers+=temp
		for x in range(num_vert):
			temp=f.readline()
			listy=temp.split()
			listy=map(float,listy)
			verts.append(listy)
		faces=f.read()
	return headers,np.array(verts),faces

def writeout(headers,verts,faces,path):
	with open(path,'wb') as f:
		vertstring=''
		for vert in verts:
			temp=vert.tolist()
			for point in temp:
				vertstring+=str(point)+' '
			vertstring+='\r\n'
		f.write(headers+vertstring+faces)



headers,verts,faces=extract_verts(INPATH)
fid=verts[np.array([43966,25067,21010,40862,6649,4197])]
ctrl=np.zeros((len(fid),3))
# print ctrl
# print fid
ctrl[0]=fid[0]+np.array([.1,0,0])
ctrl[1]=fid[1]-np.array([.1,0,0])
ctrl[2]=fid[2]-np.array([.12,0,0])
ctrl[3]=fid[3]+np.array([.12,0,0])
ctrl[4]=fid[4]
ctrl[5]=fid[5]
outverts=TPS(fid,ctrl,verts)
writeout(headers,outverts,faces,OUTPATH)