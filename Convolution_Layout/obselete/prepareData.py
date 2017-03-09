import os
import numpy as np 
#import pyqtgraph as pg 

#**************************************************************
#Load data
#**************************************************************
# gather first 600 raw data from each class
def process_data(clss, img_px_size , chunk_size, objs, base):
    cntr = 0
    new_slices = []
    for obj in objs:
    	path = base + obj
    	if (obj[2] == str(clss) and cntr < chunk_size):
    		cntr += 1
    		volume = np.fromfile(path, dtype = 'int8')
    		slices = volume.reshape([32, 32, 32])
    		new_slices.append(list(slices) ) 
    	if clss == 10:
    		if (obj[1] == str(1) and cntr < chunk_size):
	    		cntr += 1
	    		volume = np.fromfile(path, dtype = 'int8')
	    		slices = volume.reshape([32, 32, 32])
	    		new_slices.append(list(slices) ) 
    return np.array(new_slices)

# 1-hot encoding of class labels
def oneHotEncode(cls, numClasses):
	a = np.arange(numClasses)
	b = np.zeros((numClasses, numClasses+1))
	b[np.arange(numClasses), a] = 1
	label = b[cls-1][:-1]
	return label

#  some constants
home =  '/home/anusha/Anusha/scripts/05VoxNet/voxnet/'
train_dir = home + 'shapenet10_train/'
test_dir = home + 'shapenet10_test/'

IMG_SIZE_PX = 32
DEPTH = 32
trainchunk_size = 600
testchunk_size = 100
numClasses = 10
train_objs = os.listdir(train_dir)
test_objs = os.listdir(test_dir)

much_traindata = []
much_trainlabel = []

much_testdata = []
much_testlabel = []

for cls in range(1, (numClasses+1)):
    try:
        img_traindata = process_data(clss = cls, img_px_size = IMG_SIZE_PX, chunk_size = trainchunk_size, objs = train_objs, base = train_dir)
        img_testdata = process_data(clss = cls, img_px_size = IMG_SIZE_PX, chunk_size = testchunk_size, objs = test_objs, base = test_dir)
        
        #trainlabel = oneHotEncode(cls, numClasses)
        trainlabel = [cls-1] * trainchunk_size
        testlabel = [cls-1] * testchunk_size
        
        much_traindata.append(list(img_traindata))
        much_trainlabel.append(trainlabel)
        much_testdata.append(list(img_testdata))
        much_testlabel.append(testlabel)
        
    except KeyError as e:
        print('This is unlabeled data!')

np.save('much_traindata-{}-{}-{}.npy'.format(IMG_SIZE_PX, IMG_SIZE_PX, DEPTH), much_traindata)
np.save('much_trainlabel.npy' , much_trainlabel)
np.save('much_testdata-{}-{}-{}.npy'.format(IMG_SIZE_PX, IMG_SIZE_PX, DEPTH), much_testdata)
np.save('much_testlabel.npy' , much_testlabel)

listOfTrainData = np.load('much_traindata-32-32-32.npy')
trainData = [item for sublist in listOfTrainData for item in sublist] #print (len(trainData)) # 5400
trainDataa = [ x for y in trainData for x in y] #print(len(trainDataa)) #172800...
X_train = np.reshape(trainDataa, (6000,32,32,32))  #print(X_train.shape) #(5400, 32, 32, 32)

listOfTestData = np.load('much_testdata-32-32-32.npy')
testData = [item for sublist in listOfTestData for item in sublist] #print (len(testData)) # 900
testDataa = [ x for y in testData for x in y] #print(len(testDataa)) #28800...
X_test = np.reshape(testDataa, (1000,32,32,32))  #print(X_test.shape) #(1000, 32, 32, 32)


listOfTrainLabel = np.load('much_trainlabel.npy') #print(listOfTrainLabel.shape) # (10, 600)
y_train = np.reshape(listOfTrainLabel, (6000,1))

listOfTestLabel = np.load('much_testlabel.npy') #print(listOfTrainLabel.shape) # (10, 600)
y_test = np.reshape(listOfTestLabel, (1000,1))




#**************************************************************











"""
print (train_data.shape, valid_data.shape ) # (8,), (2,)
print (len(train_data[1])) #600

print (train_data[1][:].shape) #(600, 32, 32, 32)

# convert it into (4800, 32, 32, 32)

traninData = np.ravel(train_data)
print (traninData[1].shape)#(600, 32, 32, 32)
print(traninData.shape) # (8,)

td = [item for sublist in listOflist for item in sublist] 

print(len(td)) #4800
print(td[1].shape) #(32, 32, 32)
"""









 
#input('click to continue')
"""
# shapenet10
class_id_to_name = {
    "1": "bathtub",
    "2": "bed",
    "3": "chair",
    "4": "desk",
    "5": "dresser",
    "6": "monitor",
    "7": "night_stand",
    "8": "sofa",
    "9": "table",
    "10": "toilet"
}
class_name_to_id = { v : k for k, v in class_id_to_name.items() }

class_labels = set(class_id_to_name.values())

print (class_name_to_id, class_labels)
# end of shapenet10


# Some constants

home =  '/home/anusha/Anusha/scripts/05VoxNet/voxnet/'
train_dir = home + 'shapenet10_train/'

objects = os.listdir(train_dir)
#print (len(objects)) # 47892

# read data

volume = np.fromfile(train_dir + '/001.bathtub_000000001.001.raw',dtype='int8')
print (volume.shape)
v = volume.reshape([32, 32, 32])
pg.image(v)


train_data = []

for obj in objects:
	path = train_dir + obj
	volume = np.fromfile(path, dtype = 'int8')
	#print (volume.shape) #32768
	v = volume.reshape([32, 32, 32])
	#pg.image(v)
	class_label = obj[2]
	train_data.append([v, class_label])

np.save('trainData-{}-{}-{}.npy'.format(32,32,32), train_data)

"""
#**************************************************************
