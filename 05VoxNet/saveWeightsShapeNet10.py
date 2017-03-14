'''Train a simple deep CNN on the 3d shapeNet images dataset.
Visualization of the 2d filters of VGG16: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
'''

from __future__ import print_function
import os
import time
import h5py #conda install -c anaconda h5py=2.6.0y
import numpy as np
import tensorflow as tf
from scipy.misc import imsave
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Dense, Dropout, Activation, Flatten


K.set_learning_phase(0) # solves Issues in Keras model loading in Tensorflow Serving 

#**************************************************************
#misc functions
#**************************************************************
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#**************************************************************
#define some constants
#**************************************************************
# dimensions of the generated pictures for each filter.
img_width = 32
img_height = 32
img_depth = 32

# the name of the layer we want to visualize (see model definition below)
layer_name = 'convolution3d_4'

# number of convolutional filters to use at each layer
nb_filters = [32, 32]

# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [3, 3]

# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [5, 3]

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

"""
print('X_train shape:', X_train.shape)    #(6000, 32, 32, 32)
print(X_train.shape[0], 'train samples')  #6000 train samples
print(X_test.shape[0], 'test samples')    #1000 test samples
print('y_train shape:', y_train.shape)    #y_train shape: (6000, 1)
print('y_test shape:', y_test.shape)      #y_test shape: (1000, 1)
print('Y_train shape:', Y_train.shape)    #Y_train shape: (6000, 10)
print('Y_test shape:', Y_test.shape)      #Y_test shape: (1000, 10)
"""

#**************************************************************
#define the model architecture (with reference to shapenet10)
#**************************************************************
model = Sequential()

model.add(Convolution3D(nb_filters[0],kernel_dim1=nb_conv[0], kernel_dim2=nb_conv[0], kernel_dim3=nb_conv[0], border_mode='same',
                        activation='relu', dim_ordering='tf', input_shape=(1, img_width, img_height, img_depth),))
model.add(Dropout(0.2))
model.add(Convolution3D(nb_filters[1],kernel_dim1=nb_conv[1], kernel_dim2=nb_conv[1], kernel_dim3=nb_conv[1], border_mode='same',
                        activation='relu', dim_ordering='tf'))
model.add(MaxPooling3D(pool_size=(nb_pool[1], nb_pool[1], nb_pool[1]), border_mode='same', dim_ordering='tf')) #(strides=None)
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(16, init='normal', activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(nb_classes, init='normal'))
model.add(Activation('softmax'))

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


"""
SGD:
Epoch 200/200
6000/6000 [==============================] - 6s - loss: 0.3663 - acc: 0.8787 - val_loss: 0.6961 - val_acc: 0.8430


RMSprop:
Epoch 200/200
6000/6000 [==============================] - 5s - loss: 0.1009 - acc: 0.9815 - val_loss: 1.8322 - val_acc: 0.8330

"""