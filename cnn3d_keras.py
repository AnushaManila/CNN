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
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
#conda install -c anaconda basemap=1.0.7

from operator import truediv
import pyqtgraph as pg
#conda install -c anaconda pyqtgraph=0.9.10

from functools import reduce


K.set_learning_phase(0) # solves Issues in Keras model loading in Tensorflow Serving 

#**************************************************************
#misc functions
#**************************************************************
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x // (K.sqrt(K.mean(K.square(x))) + 1e-5)

# util function to convert a tensor into a valid image
def deprocess_image(x):
  # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x //= (x.std() + 1e-5)
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
layer_name = 'convolution3d_2'

batch_size = 5
nb_classes = 10
nb_epoch = 150
WEIGHTS_FNAME = 'shapenet10_cnn3_weights_epoch150_rmsprop.h5'


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

"""
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution3d_1 (Convolution3D)  (None, 1, 32, 32, 32) 128032      convolution3d_input_1[0][0]      
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1, 32, 32, 32) 0           convolution3d_1[0][0]            
____________________________________________________________________________________________________
convolution3d_2 (Convolution3D)  (None, 1, 32, 32, 32) 27680       dropout_1[0][0]                  
____________________________________________________________________________________________________
maxpooling3d_1 (MaxPooling3D)    (None, 1, 11, 11, 32) 0           convolution3d_2[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 3872)          0           maxpooling3d_1[0][0]             
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 3872)          0           flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 16)            61968       dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 16)            0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 10)            170         dropout_3[0][0]                  
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 10)            0           dense_2[0][0]                    
====================================================================================================
Total params: 217,850
Trainable params: 217,850
Non-trainable params: 0
"""


model = Sequential()

model.add(Convolution3D(32, kernel_dim1 = 5, kernel_dim2 = 5, kernel_dim3 = 5, border_mode = 'same',
                        activation ='relu', dim_ordering ='tf', input_shape = (1, img_width, img_height, img_depth)))
model.add(Dropout(0.2))
model.add(Convolution3D(32, kernel_dim1 = 3, kernel_dim2 = 3, kernel_dim3 = 3, border_mode = 'same',
                        activation ='relu', dim_ordering ='tf'))
model.add(MaxPooling3D(pool_size = (3, 3, 3), border_mode ='same', dim_ordering ='tf')) #(strides=None)
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(16, init='normal', activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(nb_classes, init='normal'))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
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
    history = LossHistory()
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              verbose=1,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[history])
    model.save_weights(WEIGHTS_FNAME)
    print ('history.losses:', history.losses)

model.summary()


#**************************************************************
# gradient ascent for visualization of the filters
#**************************************************************

input_img = model.input

# get the symbolic outputs of each "key" layer
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
print (layer_dict)


kept_filters = []
check = []
for filter_index in range(0, 31):
    # we only scan through the first few filters,
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    #print (layer_output) # Tensor("Relu_1:0", shape=(?, 1, 32, 32, 32), dtype=float32)
    
    if K.image_dim_ordering() == 'th':
        loss = K.mean(layer_output[:, filter_index, :, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, :, filter_index])
    #print ('Loss:', loss) #Loss: Tensor("Mean_64:0", shape=(), dtype=float32)

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    
    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    input_img_data = np.random.random((1, 1, img_width, img_height, img_depth))
    input_img_data = (input_img_data - 0.5) * 20 + 32
    #print (input_img_data.shape) # (1, 1, 32, 32, 32)
    #print (input_img_data[0].shape) # (1, 32, 32, 32)

    # we run gradient ascent for 20 steps
    for i in range(200):
        loss_value, grads_value = iterate([input_img_data]) # loss is a single value, grads is an array
        
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
np.save('kept_filters5-5.npy', kept_filters)




#**************************************************************
#
#**************************************************************

