import numpy as np
import h5py
import math
import sys
import argparse
from PIL import Image
import matplotlib.pyplot as plt


def array2img(array,array_max):
	#array_max = array.max()
	array = 255 - array * 255 / array_max
	#print(array.max())
	img = Image.fromarray(array)
	return img

def readHDF5(file):
	f = h5py.File(file, 'r')
	data = f["data"]
	return data

def plotSignalAndNoise(signal, noise, title=["signal", "noise", "signal+noise"], 
	out_name="figs/s+n.png"):

	signal_and_noise = signal + noise
	array_max = signal_and_noise.max()

	signal_img = array2img(signal, array_max)
	noise_img = array2img(noise, array_max)
	signal_and_noise_img = array2img(signal_and_noise, array_max)

	plt.figure(figsize=(24,12))

	ax = plt.subplot(1,3,1)
	ax.set_title(title[0])
	ax.imshow(signal_img)

	ax = plt.subplot(1,3,2)
	ax.set_title(title[1])
	ax.imshow(noise_img)

	ax = plt.subplot(1,3,3)
	ax.set_title(title[2])
	ax.imshow(signal_and_noise_img)

	plt.savefig(out_name)

	plt.close()

	return True

def getTitle(signal, noise, noise_name = '...'):
	title = ["signal : positron\ntotalPE = ", "noise : NOISE_NAME\ntotalPE = ", 
	"signal + noise\ntotalPE = "]
	title[0] += str(int(signal.sum()))
	title[1] += str(int(noise.sum()))
	title[1] = title[1].replace('NOISE_NAME', noise_name)
	title[2] += str(int(signal.sum() + noise.sum()))

	return title


if __name__ == "__main__":
	pos_path = "/hpcfs/juno/junogpu/yuansc/BackgroundRemove/dataset/pos/positron_skim_dataset_batch0_N5000.h5"
	#u238_path = "/hpcfs/juno/junogpu/yuansc/BackgroundRemove/dataset/noi/u238/u238_skim_dataset_batch0_N5000.h5"
	th232_path = "/hpcfs/juno/junogpu/yuansc/BackgroundRemove/dataset/noi/Acrylic_Th232/Acrylic_Th232_dataset_batch0_N5000.h5"

	pos = readHDF5(pos_path)
	#u238 = readHDF5(u238_path)
	th232 = readHDF5(th232_path)

	num = 30

	for i in range(num):
		#if u238[i].sum()==0 or th232[i].sum()==0:

		print("drawing fig " + str(i) + " ...")
		#title_u238 = getTitle(pos[i], u238[i], noise_name='U238')
		#plotSignalAndNoise(pos[i], u238[i] ,title=title_u238, out_name="figs/U238_"+str(i))

		title_th232 = getTitle(pos[i], th232[i], noise_name='TH232')
		plotSignalAndNoise(pos[i], th232[i] ,title=title_th232, out_name="figs/TH232_"+str(i))




