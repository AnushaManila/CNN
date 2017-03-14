import os
import numpy as np 
import pandas as pd #conda install pandas 

#**************************************************************
#Load data
#**************************************************************

def processdata(wholeImage, annotatePath):

    # get the annotations
    df = pd.read_csv(annotatePath)
    df = np.array(df)

    # get the row count
    rowCount = df.shape[0]

    # access vessel volume coordinates (x, y, z) 
    volX = df[:,0]
    volY = df[:,1]
    volZ = df[:,2]

    # crop (x, y, z) to (32, 32, 32) from the centre
    trainData = []

    for i in range(rowCount):

        volX_center, volY_center, volZ_center = int (volX[i:(i+1)] / 2), int (volY[i:(i+1)] / 2), int (volZ[i:(i+1)] / 2)
        
        # 16 voxels left and 16 voxels right to the centre of the voxel ==> 32 voxels
        volX32_start, volX32_end = (volX_center-16), (volX_center+16) 
        volY32_start, volY32_end = (volY_center-16), (volY_center+16)
        volZ32_start, volZ32_end = (volZ_center-16), (volZ_center+16)

        vol32 = wholeImage[volX32_start:volX32_end, volY32_start:volY32_end, volZ32_start:volZ32_end]  # (vol32.shape) --> (32,32,32)
        
        trainData.append(np.array(vol32))
        

    trainData = np.array(trainData)

    """
    if rowCount == 276:
        print (trainData.shape)
    if rowCount == 289:
        print (trainData.shape)
    if rowCount == 314:
        print (trainData.shape)
    """
    # get the label values for each voxel
    trainL = df[:,3] # (276,)
    trainLabel = trainL.reshape([rowCount,1])  # (276,1)


    return trainData, trainLabel



#  some constants
home =  '/home/anusha/Anusha/scripts/09VisuCT_CNN/VESSEL12/'
data_dir = home + 'Scans/'
annotate_dir = home + 'Annotations/'

rawPath1 =  data_dir + 'VESSEL12_21.raw'
annotatePath1 = annotate_dir + 'VESSEL12_21_Annotations.csv'

rawPath2 =  data_dir + 'VESSEL12_22.raw'
annotatePath2 = annotate_dir + 'VESSEL12_22_Annotations.csv'

rawPath3 =  data_dir + 'VESSEL12_23.raw'
annotatePath3 = annotate_dir + 'VESSEL12_23_Annotations.csv'


# read the data 'raw' files
volume1 = np.fromfile(rawPath1, dtype = 'int16')
wholeImage1 = volume1.reshape([512, 512, 459])

volume2 = np.fromfile(rawPath2, dtype = 'int16')
wholeImage2 = volume2.reshape([512, 512, 448])

volume3 = np.fromfile(rawPath3, dtype = 'int16')
wholeImage3 = volume3.reshape([512, 512, 418])


#print (wholeImage2.shape)
#print (wholeImage3.shape)

data1, label1 = processdata(wholeImage1, annotatePath1)
data2, label2 = processdata(wholeImage2, annotatePath2)
data3, label3 = processdata(wholeImage3, annotatePath3)

#print (data2.shape)
#print (data3.shape)

# save the train data
np.save('data1-{}-{}-{}.npy'.format(32, 32, 32), data1)
np.save('label1.npy' , label1)
np.save('data2-{}-{}-{}.npy'.format(32, 32, 32), data2)
np.save('label2.npy' , label2)

np.save('data3-{}-{}-{}.npy'.format(32, 32, 32), data3)
np.save('label3.npy' , label3)


# load the data
validData, TrainData1, TrainData2 = np.load('data1-32-32-32.npy'), np.load('data2-32-32-32.npy'), np.load('data3-32-32-32.npy'),
#print (validData.shape, TrainData1.shape, TrainData2.shape) #(276, 32, 32, 32) (289, 32, 32, 32) (314,)
validLabel, TrainLabel1, TrainLabel2 = np.load('label1.npy'), np.load('label2.npy'), np.load('label3.npy')


temp = []
print (TrainData2.shape, TrainData2[0].shape)
for i in range(314):
    temp.append(TrainData2[0])
print (np.array(temp).shape, TrainData1.shape)

data2_3 = np.concatenate((TrainData1, temp), axis=0)
print (data2_3.shape)
np.save('data2_3-{}-{}-{}.npy'.format(32, 32, 32), data2_3)

label2_3 = np.concatenate((TrainLabel1, TrainLabel2), axis=0)
print(label2_3.shape)
np.save('label2_3.npy' , label2_3)



