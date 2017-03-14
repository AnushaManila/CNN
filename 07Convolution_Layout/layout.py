from __future__ import print_function

import os
import time
import h5py #conda install -c anaconda h5py=2.6.0y
import numpy as np
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from scipy import ndimage
import pyqtgraph as pg
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 

# method to read hdf5 file
def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape))
    finally:
        f.close()

# user defined 3D cnn model
def my_small_model():
	model = Sequential()

	model.add(Convolution3D(32, kernel_dim1 = 5, kernel_dim2 = 5, kernel_dim3 = 5, border_mode='same',
	                        activation='relu', dim_ordering='tf', input_shape=(1, 32, 32, 32)))
	model.add(Dropout(0.2))
	model.add(Convolution3D(32, kernel_dim1 = 3, kernel_dim2 = 3, kernel_dim3 = 3, border_mode='same',
	                        activation='relu', dim_ordering='tf'))
	model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same', dim_ordering='tf')) #(strides=None)
	model.add(Flatten())
	model.add(Dropout(0.3))
	model.add(Dense(16, init='normal', activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(10, init='normal'))
	model.add(Activation('softmax'))
	return model

#define the model with training
def load_model():
	#**************************************************************
	#define some constants
	#**************************************************************

	batch_size = 5
	nb_classes = 10
	nb_epoch = 200
	WEIGHTS_FNAME = 'shapenet10_cnn3_weights_epoch200_RMSprop.h5'

	#**************************************************************
	#Load and prepare the data in proper dimensions
	#**************************************************************
	listOfTrainData = np.load('much_traindata-32-32-32.npy')
	trainData = [item for sublist in listOfTrainData for item in sublist] #print (len(trainData)) # 6000
	trainDataa = [ x for y in trainData for x in y] #print(len(trainDataa)) #172800...
	X_train = np.reshape(trainDataa, (6000,1,32,32,32)) 

	listOfTestData = np.load('much_testdata-32-32-32.npy')
	testData = [item for sublist in listOfTestData for item in sublist] #print (len(testData)) # 1000
	testDataa = [ x for y in testData for x in y] #print(len(testDataa)) #28800...
	X_test = np.reshape(testDataa, (1000,1,32,32,32))  

	listOfTrainLabel = np.load('much_trainlabel.npy') #print(listOfTrainLabel.shape) # (10, 600)
	y_train = np.reshape(listOfTrainLabel, (6000,1))

	listOfTestLabel = np.load('much_testlabel.npy') #print(listOfTrainLabel.shape) # (10, 600)
	y_test = np.reshape(listOfTestLabel, (1000,1))

	# Convert class vectors to binary class matrices.
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)

	#**************************************************************
	#define the model architecture (with reference to shapenet10)
	#**************************************************************

	model=my_small_model()

	# Let's train the model using RMSprop
	model.compile(loss='categorical_crossentropy',
	              optimizer='RMSprop',
	              metrics=['accuracy'])

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255

	if True and os.path.exists(WEIGHTS_FNAME):
	    # Just change the True to false to force re-training
	    print('Loading existing weights')
	    model.load_weights(WEIGHTS_FNAME)
	else:
	    model.fit(X_train, Y_train,
	              batch_size=batch_size,
	              nb_epoch=nb_epoch,
	              verbose=1,
	              validation_data=(X_test, Y_test),
	              shuffle=True)
	    model.save_weights(WEIGHTS_FNAME)
	model.summary()
	return model

# read the raw data file
def readRaw(imgPath):
	img_width = 32
	img_height = 32
	img_depth = 32

	volume = np.fromfile(imgPath, dtype = 'int8')
	slices = volume.reshape([img_width, img_height, img_depth])
	return np.array(slices)

#**************************************************************
#3D plotting the scan: 
#**************************************************************
def plot_3d(image, threshold=0):
    
        
    #extract a 2D surface mesh from a 3D volume
    verts, faces = measure.marching_cubes(image, threshold)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70) #A collection of 3D polygons


    face_color = [0.35, 0.35, 0.65]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])

    plt.show()



def main():
	# provide the input file locations
	data_path = '/home/anusha/Anusha/scripts/07Layout/data/'
	img_path = data_path + '002.bed_000000939.004.raw'
	weights_path = data_path + 'shapenet10_cnn3_weights_epoch200_RMSprop.h5'

	# get the img, model and the weights
	img = readRaw(img_path) #print (img.shape): (32,32,32)
	model = my_small_model()
	model.load_weights(weights_path, by_name=False)

	#for layer in model.layers:
	    #weights = layer.get_weights() # list of numpy arrays
	w = model.layers[0].get_weights()  # for weight

	weights = w[0][:,:,:,1,31] # w[0].shape: (5, 5, 5, 32, 32) and (weights.shape) = (5,5,5)
	biases = w[1] # shape: (32,)

	# do convolution
	conv3d = ndimage.convolve(img, 10*weights) # weights are too small, multiply by 10 to visualize
	print (conv3d.shape)
	#np.ndarray.dump(np.array(conv3d),'conv3d.txt')
	# visualize the output
	pg.image(conv3d)
	#pg.image(10*weights)

	#plot_3d(conv3d) # set the threshold carefully
	#**************************************************************
	#
	#**************************************************************
	input('  ')

main()
