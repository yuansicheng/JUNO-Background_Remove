import sys
sys.path.append("/hpcfs/juno/junogpu/yuansc/my_lib")

import initialise_logging
import logging

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD

def createModel(data_train_shape):
	logging.info('creating model!')

	model = Sequential()
	ki = 'random_normal'

	model.add(Conv2D(64, (3, 3), strides=(1,1), activation='relu', \
    padding='same', input_shape=(data_train_shape), kernel_initializer=ki))
	model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=ki))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(BatchNormalization(axis=1))

	model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=ki))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(BatchNormalization(axis=1))

	#model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=ki))
	#model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=ki))
	model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=ki))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization(axis=1))

	model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=ki))
	#model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=ki))
	model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=ki))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(BatchNormalization(axis=1))

	model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=ki))
	#model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=ki))
	#model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=ki))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization(axis=1))

	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	#model.add(Dense(1024, activation='relu'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.1))
	model.add(Dense(64, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	model.add(Dense(2, activation='softmax'))

	sgd = SGD(lr=0.001)#, decay=1e-6)#, momentum=0.9#, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])

	return model