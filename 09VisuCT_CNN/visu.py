import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 

#**************************************************************
#3D plotting the scan
#**************************************************************
def plot_3d(image, threshold=-300):
    
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

#**************************************************************
# final visualization of the filters
#**************************************************************
savedFilter = np.ravel(np.array(np.load('kept_filters.npy'), dtype='int32'))
print ('savedFilter:', savedFilter.shape) #(753664,) 32*32*32*23 = 753664
sliced32 = savedFilter[-32768:] # 23 such
 
filterVolume32 = sliced32.reshape([32,32,32]) 
#Display 3D data
pg.image(filterVolume32)

print (np.amax(sliced32))

plot_3d(filterVolume32, 100)




input('  ')
