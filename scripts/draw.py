from PIL import Image
import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
def array2img(array):
	#array_max = array.max()
	array = 255 - array * 255 / np.max(array)
	#print(array.max())
	img = Image.fromarray(array)
	return img

def saveImg(img, path):
	img = img.convert('L')
	img.save(path)

def readHDF5(file, key='data'):
    print('readHDF5: ', file)
    f = h5py.File(file, 'r')
    data = f[key]
    return data

file = [f for f in os.listdir('.') if '.h5' in f][0]
n=0
for d in readHDF5(file):
	if n>10:
		break
	img = array2img(d[:,:,0])
	saveImg(img,str(n)+'_0.png')
	img = array2img(d[:,:,1])
	saveImg(img,str(n)+'_1.png')
	n+=1
