import numpy as np

def parse_file(path):
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
