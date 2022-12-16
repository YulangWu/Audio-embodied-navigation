import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import os
import random
import cv2


sampling = 8
amplifier = 2
#Rs_int_forward_illumination_homo250_250.npy
illuminationname = 'Rs_int_output/Rs_int_forward_illumination273_347.npy'
illumination = np.load(illuminationname)

floor_plan = mpimg.imread('scenes/Rs_int/layout/floor_trav_no_obj_0.png')

floor_plan = np.transpose(floor_plan)
nx = len(floor_plan)
nz = nx

nz,nx = int(nz/amplifier),int(nx/amplifier)
for i in range(nx):
    for j in range(nz):
        floor_plan[i,j] = floor_plan[i*amplifier,j*amplifier]

floor_plan = floor_plan[:nx,:nz]

flow = np.multiply(floor_plan,illumination)

gx = np.zeros((nx,nz))
gz = np.zeros((nx,nz))
angle_map = np.zeros((nx,nz))
norm = np.zeros((nx,nz))
#Get gradient
for i in range(nx-1):
    if floor_plan[i,nx-1] == 1:
        gx[i,nx-1] = illumination[i+1,nx-1]-illumination[i,nx-1]
for j in range(nz-1):
    if floor_plan[nz-1,j] == 1:
        gz[nz-1,j] = illumination[nz-1,j+1]-illumination[nz-1,j]

for i in range(nx-1):
    for j in range(nz-1):
        if floor_plan[i,j] == 1:
            gx[i,j] = illumination[i+1,j]-illumination[i,j]
            gz[i,j] = illumination[i,j+1]-illumination[i,j]
            norm[i,j] = np.sqrt(gx[i,j]**2 + gz[i,j]**2)
            gx[i,j] /= norm[i,j]/100
            gz[i,j] /= norm[i,j]/100

for i in range(nx):
    for j in range(nz):
        angle_map[i,j] = np.arctan2(gz[i,j],gx[i,j])/np.pi*180
# min_val = np.min(illumination)
# print(min_val)
# illumination /= min_val*10
# print(illumination)

for i in range(nx):
    for j in range(nz):
        illumination[i,j] = np.log10(illumination[i,j])
        norm[i,j] = np.log10(norm[i,j])

#downsample and normalization
ggx = np.zeros((int(nx/sampling),int(nz/sampling)))
ggz = np.zeros((int(nx/sampling),int(nz/sampling)))
floor_plan2 = np.zeros((int(nx/sampling),int(nz/sampling)))
for i in range(int(nx/sampling)):
    for j in range(int(nz/sampling)):
        ggx[i,j] = gx[i*sampling,j*sampling]
        ggz[i,j] = gz[i*sampling,j*sampling]
        floor_plan2[i,j] = floor_plan[i*sampling,j*sampling]


plt.figure(1)
plt.subplot(2,2,1)
plt.imshow(floor_plan)
plt.xlabel('z direction')
plt.ylabel('x direction')
plt.gca().invert_yaxis()

plt.subplot(2,2,1)
plt.imshow(np.multiply(illumination,floor_plan))
# plt.clim(-3.2,1.2)
plt.xlabel('z direction')
plt.ylabel('x direction')
plt.title('illumination')
plt.gca().invert_yaxis()
# plt.set_cmap('Reds')

plt.subplot(2,2,2)
plt.imshow(np.multiply(angle_map,floor_plan))
plt.clim(-180,180)
plt.xlabel('z direction')
plt.ylabel('x direction')
plt.title('angle map')
plt.gca().invert_yaxis()
# plt.set_cmap('Reds')

plt.subplot(2,2,3)
plt.imshow(np.multiply(norm,floor_plan))
plt.clim(-4, 4)
plt.xlabel('z direction')
plt.ylabel('x direction')
plt.title('grad_norm')
plt.gca().invert_yaxis()
plt.set_cmap('seismic')


ggx = np.reshape(ggx,(int(nx/sampling)*int(nz/sampling),1))
ggz = np.reshape(ggz,(int(nx/sampling)*int(nz/sampling),1))
floor_plan2 = np.reshape(floor_plan2,(int(nx/sampling)*int(nz/sampling),1))

xline = np.linspace(0,nx-1,int(nx/sampling))
zline = np.linspace(0,nz-1,int(nz/sampling))
xx,zz = np.meshgrid(xline,zline)
xx = np.reshape(xx,(int(nx/sampling)*int(nz/sampling),1))
zz = np.reshape(zz,(int(nx/sampling)*int(nz/sampling),1))

x_pos = []
y_pose = []
x_direction = []
y_direction = []

count=0
for i in range(int(nx/sampling)*int(nz/sampling)):
    if floor_plan2[i] == 1:
        count += 1
        x_pos.append(xx[i])
        y_pose.append(zz[i])
        x_direction.append(-ggx[i])
        y_direction.append(-ggz[i])


# x = np.linspace(0,nx-1,nx)
# y = np.linspace(0,nz-1,nz)
# xx,yy = np.meshgrid(x,y)
# fig = plt.figure(2)
# sub = fig.add_subplot(111,projection='3d')
# subf=sub.plot_surface(xx,yy,illumination,color='blue')
# plt.show()



plt.subplot(2,2,4)
#change x and y direction because for plot, x is horizontal y is vertical
plt.quiver(x_pos,y_pose,y_direction,x_direction)
plt.title('quiver')
# plt.quiver(xx,zz,gz,gx)
plt.axis('equal')
# plt.gca().invert_yaxis() DO NOT USE IT!
plt.xlabel('z direction')
plt.ylabel('x direction')
plt.show()



x = np.multiply(illumination,floor_plan) 
np.savetxt('illumination.dat',x,fmt="%8.7f")       
    