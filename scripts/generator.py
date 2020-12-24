import sys
sys.path.append("/hpcfs/juno/junogpu/yuansc/my_lib")

import initialise_logging
import logging

import os

import h5py
import numpy as np 
import random
import copy

from tensorflow.keras.utils import to_categorical

def readHDF5(file):
    logging.info('readHDF5: '+ file)
    data = h5py.File(file, 'r')
    return data

def getH5Files(path, file_num=None):
    logging.info('getH5Files: '+ path)
    h5_files = [path+p for p in os.listdir(path) if '.h5' in p]
    if file_num:
        try:
            h5_files = h5_files[:file_num]
        except:
            logging.warning('getH5Files: not so much h5 files, already select all h5 files')
    return h5_files

def concatData(data_list):
    #logging.debug('concatData ...')
    try:
        result = data_list[0]
    except:
        logging.error('concatData: no data in data_list')
        return False
    for data in data_list[1:]:
        result = np.vstack((result, data))
    return result

def cut(data, metric, condition=[-float('inf'), float('inf')]):
	if condition[0]>condition[1]:
		logging.error('condition[0]>condition[1]')
		return data
	return data[np.where((metric>=condition[0]) & (metric<=condition[1]))]

def concatenate(data, new_axis=True, axis_num=3):
	if len(set([len(d) for d in data]))>1:
		logging.error(len(set([d.shape for d in data]))>1)
		return False
	if new_axis:
		data = [d[:,:,:,np.newaxis] for d in data]
	return np.concatenate(data, axis=axis_num)

def mix(sig, noi):
	#logging.debug('sig.shape:'+str(sig.shape))
	#logging.debug('noi.shape:'+str(noi.shape))
	sig[:,:,0] = sig[:,:,0] + noi[:,:,0]
	#logging.debug('np.sum(sig[:,:,1]), before mix:'+str(np.sum(sig[:,:,1])))

	#logging.debug('np.sum(np.maximum(sig[:,:,1],noi[:,:,1])):'+str(np.sum(np.maximum(sig[:,:,1],noi[:,:,1]))))

	sig[:,:,1] = np.where(sig[:,:,0]*noi[:,:,0]==0, \
			np.maximum(sig[:,:,1],noi[:,:,1]), \
			np.minimum(sig[:,:,1],noi[:,:,1]))
	#logging.debug('np.sum(sig[:,:,1]), after mix:'+str(np.sum(sig[:,:,1])))

	return sig

def createDataset(sig_file, noi_file, size=5000, \
	cut_conditions={}, \
	noi_ratio=0.5):

	sig_data = readHDF5(sig_file)
	noi_data = readHDF5(noi_file)
	
	sig_pmt_id = sig_data['pmt_hit'][:]
	sig_hit_time = sig_data['first_hit_time'][:]
	noi_pmt_id = noi_data['pmt_hit'][:]
	noi_hit_time = noi_data['first_hit_time'][:]

	'''
	logging.debug(type(sig_pmt_id))
	logging.debug(type(sig_hit_time))
	logging.debug(type(noi_pmt_id))
	logging.debug(type(noi_hit_time))
	'''

	sig_2c = concatenate([sig_pmt_id, sig_hit_time])
	noi_2c = concatenate([sig_pmt_id, sig_hit_time])

	logging.info('sig_2c.shape, before cut:'+str(sig_2c.shape))
	logging.info('noi_2c.shape, brfore cut:'+str(noi_2c.shape))


	if 'sig_r_cut' in cut_conditions.keys():
		sig_2c = cut(sig_2c, sig_data['init_r'][:], condition=cut_conditions['sig_r_cut'])
	if 'sig_npe_cut' in cut_conditions.keys():
		sig_2c = cut(sig_2c, sig_data['npe'][:], condition=cut_conditions['sig_npe_cut'])
	if 'noi_r_cut' in cut_conditions.keys():
		noi_2c = cut(noi_2c, noi_data['init_r'][:], condition=cut_conditions['noi_r_cut'])
	if 'noi_npe_cut' in cut_conditions.keys():
		noi_2c = cut(noi_2c, noi_data['npe'][:], condition=cut_conditions['noi_npe_cut'])


	logging.info('sig_2c.shape, after cut:'+str(sig_2c.shape))
	logging.info('noi_2c.shape, after cut:'+str(noi_2c.shape))

	sig_len = sig_2c.shape[0]
	noi_len = noi_2c.shape[0]

	labels = np.zeros((size,1))
	data = np.zeros((size, sig_2c.shape[1], sig_2c.shape[2], sig_2c.shape[3]))

	noi_index = 0
	for i in range(size):
		if random.random() < noi_ratio:
			data[i] = mix(sig_2c[i%sig_len], noi_2c[noi_index%noi_len])
			labels[i] = 1
			noi_index += 1
		else:
			data[i] = sig_2c[i%sig_len]

	labels = to_categorical(labels, num_classes=2)

	return data, labels


def myGenerator(sig_path, noi_path, batch_size, file_num, cut_conditions={}):
	sig_files = getH5Files(sig_path, file_num=file_num)
	noi_files = getH5Files(noi_path, file_num=file_num)
	files_zip = list(zip(sig_files, noi_files))
	
	n = 0
	while 1:
		if n>100:
			return
		for z in files_zip:
			data, labels = createDataset(z[0], z[1], size=(5000//batch_size)*batch_size, cut_conditions=cut_conditions)
			for i in range(5000//batch_size):
				#logging.debug('i,5000//batch_size: '+str(i)+','+str(5000//batch_size))
				yield(data[batch_size*i: batch_size*(i+1)], labels[batch_size*i: batch_size*(i+1)])
		#return True

def createTestDataset(sig_path, noi_path, batch_size, file_num, cut_conditions={}):
	sig_files = getH5Files(sig_path, file_num=file_num)
	noi_files = getH5Files(noi_path, file_num=file_num)
	files_zip = zip(sig_files, noi_files)
	
	data_all = [createDataset(z[0], z[1], size=5000, cut_conditions=cut_conditions) for z in files_zip]
	data = concatenate([d[0] for d in data_all], new_axis=False, axis_num=0)
	labels = concatenate([d[1] for d in data_all], new_axis=False, axis_num=0)
	logging.info('test_data.shape:'+str(data.shape))
	logging.info('test_labels.shape:'+str(labels.shape))
	return data, labels





