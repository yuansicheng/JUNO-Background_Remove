import sys
sys.path.append("/hpcfs/juno/junogpu/yuansc/my_lib")

import initialise_logging
import logging

import tensorflow as tf

from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau

from model import *
from generator import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO)

	#---------------------------------------------------------------------------------------
	#args and dataset

	train_sig_path = '/hpcfs/juno/junogpu/yuansc/BackgroundRemove/dataset/pos_2c/train/'
	train_noi_path = '/hpcfs/juno/junogpu/yuansc/BackgroundRemove/dataset/noi/c14_2c/train/'
	test_sig_path = '/hpcfs/juno/junogpu/yuansc/BackgroundRemove/dataset/pos_2c/test/'
	test_noi_path =	'/hpcfs/juno/junogpu/yuansc/BackgroundRemove/dataset/noi/c14_2c/test/'

	batch_size = 128
	train_file_num = 15
	test_file_num = 1

	#cut_conditions={'noi_npe_cut':(100,float('inf'))}
	cut_conditions={}

	data_test, labels_test = createTestDataset(test_sig_path, test_noi_path, \
		batch_size, test_file_num, cut_conditions=cut_conditions )

	logging.info('data_test.shape:'+str(data_test.shape))
	logging.info('labels_test.shape:'+str(labels_test.shape))
	#logging.debug('labels_test[:10]:'+str(labels_test[:10]))

	train_gen = myGenerator(train_sig_path, train_noi_path, batch_size, train_file_num, cut_conditions=cut_conditions)
	val_gen = myGenerator(test_sig_path, test_noi_path, batch_size, test_file_num, cut_conditions=cut_conditions)

	#----------------------------------------------------------------------------------------
	#model and train
	model = createModel(data_test.shape[1:])

	logging.info('tf.test.is_gpu_available: '+str(tf.test.is_gpu_available()))
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, mode='auto')
	early_stopping = EarlyStopping(monitor='val_loss', patience=5)
	logging.info('starting training...')
	
	'''
	history = model.fit(data_train, labels_train, validation_split=0.1, \
		batch_size=64, epochs=200, shuffle=False, callbacks=[reduce_lr, early_stopping])
	'''

	history = model.fit_generator(train_gen, (5000//batch_size)*train_file_num, \
							epochs=40, callbacks=[reduce_lr, early_stopping], \
							validation_data=val_gen, validation_steps=(5000//batch_size)*test_file_num)

	logging.info('training complete...')

	logging.info('history')
	for key in history.history.keys():
		logging.info(key+':'+str(history.history[key]))

	#score = model.evaluate(data_test, labels_test, batch_size=batch_size)
	
	

