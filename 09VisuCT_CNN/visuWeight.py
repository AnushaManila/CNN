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
	model.add(Dense(2, init='normal'))
	model.add(Activation('softmax'))
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
	data_path = '/home/anusha/Anusha/scripts/09VisuCT_CNN/'
	weights_path = data_path + 'vessel12_cnn3_weights_epoch200_rmsprop.h5'

	# get the model and the weights
	print_structure(weights_path)

	model = my_small_model()
	model.load_weights(weights_path, by_name=False)

	#for layer in model.layers:
	    #weights = layer.get_weights() # list of numpy arrays
	w = model.layers[0].get_weights()  # for weight


	weights = w[0][:,:,:,31,31] # w[0].shape: (5, 5, 5, 32, 32) and (weights.shape) = (5,5,5)
	#biases = w[1] # shape: (32,)

	
	pg.image(10*w[0][:,:,:,31,31])
	pg.image(10*w[0][:,:,:,31,30])
	pg.image(10*w[0][:,:,:,30,31])

	#plot_3d(conv3d) # set the threshold carefully
	#**************************************************************
	#
	#**************************************************************
	input('  ')

main()
