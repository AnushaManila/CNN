import time
import h5py #conda install -c anaconda h5py=2.6.0y
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
#conda install -c anaconda basemap=1.0.7

from operator import truediv
import pyqtgraph as pg
#conda install -c anaconda pyqt=4.11.4

#from functools import reduce



#**************************************************************
# final visualization of the filters
#**************************************************************
savedFilter = np.ravel(np.array(np.load('kept_filters5-5.npy'))) #print ('savedFilter:', savedFilter.shape) # (40,)

print (np.shape(savedFilter))

sliced = savedFilter[1:126] # slice array of shape (27,) to be able to reshape next
print (sliced.shape)
filterVolume = sliced.reshape([5,5,5]) 
#filterVolume = np.array(filterVolume)
#print (filterVolume)
#print (np.shape(filterVolume))
pg.image(filterVolume)

#print (reduce(max, savedFilter.flat))
print ('min:', np.amin(filterVolume))



#**************************************************************
#3D plotting the scan
#**************************************************************
def plot_3d(image, threshold=33):
    
    # Position the scan upright, 
    p = image.transpose(2,1,0)
    
    #extract a 2D surface mesh from a 3D volume
    verts, faces = measure.marching_cubes(p, threshold)
    

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70) #A collection of 3D polygons


    face_color = [0.35, 0.35, 0.65]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


plot_3d(filterVolume)

#**************************************************************
#
#**************************************************************
input('  ')
