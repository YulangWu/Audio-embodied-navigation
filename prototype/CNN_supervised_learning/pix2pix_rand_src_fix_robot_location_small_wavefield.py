#####################################################
#       Acoustic source location estimation
#        The University of Texas at Dallas
#                     Yulang Wu
#                     2022.12.13
#####################################################
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import argparse
import os
import collections
import matplotlib.image as mpimg
import random

import torch #PyTorch tutorial
import torchvision #PyTorch tutorial
import torchvision.transforms as transforms #PyTorch tutorial

import matplotlib.pyplot as plt #PyTorch tutorial
import numpy as np #PyTorch tutorial

import torch.nn as nn #PyTorch tutorial
import torch.nn.functional as F #PyTorch tutorial

import torch.optim as optim #PyTorch tutorial

parser = argparse.ArgumentParser()
# model parameters
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--ngf', type=int, default=1024, help='# number of hidden layers')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
#input from pysit
parser.add_argument("--num_shots_ratio", type=float, default=0.01, help="Number of shots in the floor plan")
parser.add_argument("--ncenters", type=int, default=100, help="Number of robot centers in the floor plan")
parser.add_argument("--amplifier", type=int, default=2, help="increase the unit length by amplifier")
parser.add_argument("--dh", type=float, default=0.01, help="dh = m")
parser.add_argument("--t", type=int, default=30, help="time = ms") #25
parser.add_argument("--v_air", type=float, default=0.343, help="indoor air velocity")#
parser.add_argument("--v_wall", type=float, default=4.2, help="indoor wall velocity")
parser.add_argument("--v_outside", type=float, default=0.343, help="outside velocity (include outer wall)")
parser.add_argument("--cfl", type=float, default=1/1.414, help="temperal increment.")
parser.add_argument("--one_side", type=int, default=2, help="one_side * 2 + 1 = maximum square in robot")
parser.add_argument("--f_peak", type=int, default=0.1, help="Peak frequency")
parser.add_argument("--display", type=int, default=1000, help="for live plotting")
parser.add_argument("--order", type=int, default=2, help="spatial order.")
parser.add_argument("--n", type=int, default=8, help="spatial order.")
parser.add_argument("--max_v", type=int, default=5.0, help="Maximum velocity.")
parser.add_argument("--type1", default= 'layout/floor_trav_no_obj_0.png')
parser.add_argument("--type2", default= 'layout/floor_trav_no_door_0.png')
parser.add_argument("--type3", default= 'layout/floor_trav_no_obj_no_wall_0.png')
parser.add_argument("--model_name", default= 'Rs_int', help="output directory.")
parser.add_argument("--scenes_dir", default='scenes', help="output directory.")
parser.add_argument("--output_dir", default='output', help="output directory.")
parser.add_argument("--figs", type=bool, default=False, help="plot figures")
parser.add_argument("--cpml_len", type=int, default=0.1, help="the length of cpml in one side")
parser.add_argument("--cpml_amp", type=int, default=200, help="Maximum velocity.")

parser.add_argument("--nf", type=int, default=256, help="useless")
parser.add_argument("--izs", type=int, default=500, help="useless")
parser.add_argument("--izr", type=int, default=126, help="useless")
parser.add_argument("--nx", type=int, default=512, help="useless")
parser.add_argument("--nz", type=int, default=512, help="useless")
parser.add_argument("--CNNweights_dir", default='CNNweights', help="output directory.")
parser.add_argument("--mode", default="train", choices=["train", "test", "export"])
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs with the initial learning rate')
parser.add_argument('--robot_x', type=int, default=171, help='# of input image channels: 3 for RGB and 1 for grayscale')
parser.add_argument('--robot_z', type=int, default=253, help='# of input image channels: 3 for RGB and 1 for grayscale')
# export options
a = parser.parse_args()

if not os.path.exists(a.CNNweights_dir):
        os.makedirs(a.CNNweights_dir)

for k, v in a._get_kwargs():
        print(k, " = ", v)

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc , outer_nc, #inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            # up = [uprelu, upconv, nn.Tanh()] #for original U-net
            up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, #inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return self.model(x) #torch.cat([x, self.model(x)], 1)

class Net(nn.Module):
    def __init__(self,in_channel=299,out_channel=250000,conv_size=3):
        print('in_channel={},out_channel={},conv_size={}'.format(
            in_channel,out_channel,conv_size))

        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, a.ngf, conv_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.norm_layer = nn.InstanceNorm2d(a.ngf)
        self.norm_layer1D = nn.InstanceNorm1d(a.ngf*4)
        self.conv2 = nn.Conv2d(a.ngf, a.ngf, conv_size) #layer is 16,nx,nz
        self.conv3 = nn.Conv2d(a.ngf, a.ngf, conv_size) #layer is 16,nx,nz
        self.fc1 = nn.Linear((a.one_side*2+1)*(a.one_side*2+1)*a.ngf, a.ngf*4)
        self.fc2 = nn.Linear(a.ngf*4, a.ngf*4)
        self.fc3 = nn.Linear(a.ngf*4, out_channel)

    def forward(self, x):
        # print('Input data:',x.size())

        x = self.conv1(x)
        x = F.pad(x, (1, 1, 1, 1)) # [left, right, top, bot]
        x = F.relu(x)
        x = self.norm_layer(x)
        # print('1st conv:',x.size())

        # x = self.pool(x)
        # print('1st pool:',x.size())

        res = x
        x = self.conv2(x)
        x = F.pad(x, (1, 1, 1, 1)) # [left, right, top, bot]
        x += res
        x = F.relu(x)
        x = self.norm_layer(x)
        # print('2nd conv:',x.size())

        # x = self.pool(x)
        # print('1st pool:',x.size())

        res = x
        x = self.conv2(x)
        x = F.pad(x, (1, 1, 1, 1)) # [left, right, top, bot]
        x += res
        x = F.relu(x)
        x = self.norm_layer(x)
        # print('3nd conv:',x.size())
        
        res = x
        x = self.conv2(x)
        x = F.pad(x, (1, 1, 1, 1)) # [left, right, top, bot]
        x += res
        x = F.relu(x)
        x = self.norm_layer(x)
        # print('4th conv:',x.size())

        res = x
        x = self.conv2(x)
        x = F.pad(x, (1, 1, 1, 1)) # [left, right, top, bot]
        x += res
        x = F.relu(x)
        x = self.norm_layer(x)
        # print('5th conv:',x.size())

        res = x
        x = self.conv2(x)
        x = F.pad(x, (1, 1, 1, 1)) # [left, right, top, bot]
        x += res
        x = F.relu(x)
        x = self.norm_layer(x)
        # print('6th conv:',x.size())

        res = x
        x = self.conv2(x)
        x = F.pad(x, (1, 1, 1, 1)) # [left, right, top, bot]
        x += res
        x = F.relu(x)
        x = self.norm_layer(x)
        # print('7th conv:',x.size())

        res = x
        x = self.conv2(x)
        x = F.pad(x, (1, 1, 1, 1)) # [left, right, top, bot]
        x += res
        x = F.relu(x)
        x = self.norm_layer(x)
        # print('8th conv:',x.size())

        res = x
        x = self.conv2(x)
        x = F.pad(x, (1, 1, 1, 1)) # [left, right, top, bot]
        x += res
        x = F.relu(x)
        x = self.norm_layer(x)
        # print('9th conv:',x.size())

        res = x
        x = self.conv2(x)
        x = F.pad(x, (1, 1, 1, 1)) # [left, right, top, bot]
        x += res
        x = F.relu(x)
        x = self.norm_layer(x)
        # print('10th conv:',x.size())

        # x = self.conv2(x)
        # x = F.pad(x, (1, 1, 1, 1)) # [left, right, top, bot]
        # x = F.relu(x)
        # # print('1st conv:',x.size())

        # x = self.pool(x)
        # # print('1st pool:',x.size())

        res = x
        x = self.conv3(x)
        x = F.pad(x, (1, 1, 1, 1)) # [left, right, top, bot]
        x += res
        x = F.relu(x)
        x = self.norm_layer(x)
        # print('11th conv:',x.size())

        # print('4th conv:',x.size())

        # x = self.pool(x)
        # print('2nd pool:',x.size())

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print('flattened layer:',x.size())

        x = F.relu(self.fc1(x))
        x = self.norm_layer1D(x)
        # print('Relu(fc1):',x.size())

        x = F.relu(self.fc2(x))
        x = self.norm_layer1D(x)
        # print('Relu(fc2):',x.size())

        x = self.fc3(x)
        # print('fc3:',x.size())

        return x

def check_validity(pos,floor_plan,one_side):
    x = pos[0]
    z = pos[1]

    if np.sum(floor_plan[x-one_side:x+one_side+1,z-one_side:z+one_side+1]) == (one_side*2+1)*(one_side*2+1):
        return True
    else:
        return False

def load_scene_image():
    #load floot plan:
    floor_plan_no_furnitures = mpimg.imread(os.path.join(a.scenes_dir,os.path.join(a.model_name,a.type1)))
    floor_plan_furniture = mpimg.imread(os.path.join(a.scenes_dir,os.path.join(a.model_name,a.type2)))
    nx = len(floor_plan_no_furnitures)
    nz = nx

    floor_plan_no_furnitures.shape = nx, nz
    floor_plan_furniture.shape = nx, nz
    floor_plan_no_furnitures = np.transpose(floor_plan_no_furnitures)
    floor_plan_furniture = np.transpose(floor_plan_furniture)
    #1.2 convert to the smaller one (e.g., nx/a.amplifier ==> 0.01m to 0.01m*amplifier ) 
    nz,nx = int(nz/a.amplifier),int(nx/a.amplifier)
    for i in range(nx):
        for j in range(nz):
            floor_plan_furniture[i,j] = floor_plan_furniture[i*a.amplifier,j*a.amplifier]
            floor_plan_no_furnitures[i,j] = floor_plan_no_furnitures[i*a.amplifier,j*a.amplifier]

    floor_plan_no_furnitures = floor_plan_no_furnitures[:nx,:nz]
    floor_plan_furniture = floor_plan_furniture[:nx,:nz]
    
    free_space = [(0,0)]*(nx*nz)
    count = 0
    #create wavefield using empty room (no furniture)
    for i in range(nx):
        for j in range(nz):
            if floor_plan_furniture[i,j] == 1: #free space
                #add this free space to the list:
                free_space[count] = [i,j]
                count += 1

    #truncate the list
    free_space = free_space[0:count]
    print('size of free space = ',count)

    res = {}
    res['floor_plan_no_furnitures'] = floor_plan_no_furnitures
    res['floor_plan_furniture'] = floor_plan_furniture
    res['nx'] = nx
    res['nz'] = nz
    res['free_space_index'] = free_space

    return res

def random_zone(floor_plan_furniture, free_space, ncenters, one_side):
    robot_pos = [] #store the different index of robot position
    for i in range(ncenters):
        while True:
            j = random.randrange(0,len(free_space))
            # print(check_validity(free_space[j],floor_plan_furniture,a.one_side))
            if check_validity(free_space[j],floor_plan_furniture,one_side) and free_space[j] not in robot_pos:
                robot_pos.append([free_space[j][0],free_space[j][1]])
                break
    return robot_pos

def illumination_aug_norm(illumination, floor_plan, return_parameters = []):
    epsinol = 0.000001
    nx, nz = np.shape(illumination)
    gx = np.zeros((nx,nz))
    gz = np.zeros((nx,nz))
    angle_map = np.zeros((nx,nz))
    grad_norm = np.zeros((nx,nz))
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
                
    for i in range(nx):
        for j in range(nz):
            grad_norm[i,j] = np.sqrt(gx[i,j]**2 + gz[i,j]**2)
            angle_map[i,j] = np.arctan2(gz[i,j],gx[i,j])

    for i in range(nx):
        for j in range(nz):
            illumination[i,j] = np.log10(illumination[i,j] + epsinol)
            grad_norm[i,j] = np.log10(grad_norm[i,j] + epsinol) 

    #crop data to indoor zone
    illumination = np.multiply(illumination,floor_plan)
    gx = np.multiply(gx, floor_plan)
    gz = np.multiply(gz, floor_plan)
    angle_map = np.multiply(angle_map,floor_plan)
    grad_norm = np.multiply(grad_norm,floor_plan)

    print(np.max(illumination))
    print(np.max(angle_map))
    print(np.max(grad_norm))

    #normalize data to [-1,1]
    illumination = illumination/np.max(illumination) #- np.ones((nx,nz))
    angle_map = angle_map/np.max(np.abs(angle_map)) 
    grad_norm = grad_norm/np.max(grad_norm) #- np.ones((nx,nz))

    res = {}
    if 'illumination' in return_parameters:
        res['illumination'] = illumination
    if 'gx' in return_parameters:
        res['gx'] = gx
    if 'gz' in return_parameters:
        res['gz'] = gz
    if 'grad_norm' in return_parameters:
        res['grad_norm'] = grad_norm
    if 'angle_map' in return_parameters:
        res['angle_map'] = angle_map
        
    return res

def load_illumination(illumination_dir,floor_plan,
    return_parameters=['illumination', 'gx', 'gz', 'grad_norm', 'angle_map']):
    illumination_set = []
    for file in os.listdir(illumination_dir):
        if "illumination" in file:
            print(len(os.listdir(illumination_dir)),file)
            #read illumination
            illumination = np.load(os.path.join(illumination_dir,file))
      
            #data augmentation and normalization
            res = illumination_aug_norm(illumination,floor_plan, 
            return_parameters=['illumination', 'gx', 'gz', 'grad_norm', 'angle_map'])
            illumination_set.append(res)

            # plt.figure(1)
            # plt.subplot(2,3,1)
            # plt.imshow(floor_plan)
            # plt.title('floor_plan')
            # plt.xlabel('z direction')
            # plt.ylabel('x direction')
            # plt.set_cmap('seismic')
            # plt.gca().invert_yaxis()

            # plt.subplot(2,3,2)
            # plt.imshow(illumination_set[-1]['illumination'])
            # plt.clim(-3,3)
            # plt.xlabel('z direction')
            # plt.ylabel('x direction')
            # plt.title('illumination')
            # plt.gca().invert_yaxis()
            # # plt.set_cmap('Reds')

            # plt.subplot(2,3,3)
            # plt.imshow(illumination_set[-1]['gx'])
            # plt.clim(-3,3)
            # plt.xlabel('z direction')
            # plt.ylabel('x direction')
            # plt.title('gx')
            # plt.gca().invert_yaxis()
            # # plt.set_cmap('Reds')

            # plt.subplot(2,3,4)
            # plt.imshow(illumination_set[-1]['gz'])
            # plt.clim(-3,3)
            # plt.xlabel('z direction')
            # plt.ylabel('x direction')
            # plt.title('gz')
            # plt.gca().invert_yaxis()
            # # plt.set_cmap('Reds')

            # plt.subplot(2,3,5)
            # plt.imshow(illumination_set[-1]['angle_map'])
            # plt.clim(-3,3)
            # plt.xlabel('z direction')
            # plt.ylabel('x direction')
            # plt.title('angle_map')
            # plt.gca().invert_yaxis()
            # # plt.set_cmap('Reds')

            # plt.subplot(2,3,6)
            # plt.imshow(illumination_set[-1]['grad_norm'])
            # plt.clim(-3,3)
            # plt.xlabel('z direction')
            # plt.ylabel('x direction')
            # plt.title('grad_norm')
            # plt.gca().invert_yaxis()
            # plt.show()
            # plt.savefig(os.path.join(illumination_dir,file[:-4]))

    return illumination_set

def load_frq_wavefield_fix_window(frq_wavefield_dir,return_parameters=['real','imag','pos']):
    file_list = os.listdir(frq_wavefield_dir)
    file_list.sort()

    if 'pos' in return_parameters:
        pos = []
        for file in file_list:
            if "real_frqwavefield" in file:
                i = file.find('wavefield') + len('wavefield')
                j = file.rfind('_')
                k = file.find('.npy')
                pos.append([int(file[i:j]),int(file[j+1:k])])

                print('x={};z={}'.format(pos[-1][0],pos[-1][1]))

        #output_training_source_positions
        output_src_pos = np.array(pos)
        output_src_pos.shape = -1,1
        np.savetxt(a.mode + '_src_pos.txt',output_src_pos, delimiter=',')  

    if 'real' in return_parameters:
        real_frq_wavefield = []
        for file in file_list:
            if "real_frqwavefield" in file: #load both real and imag
                print(file)
                wavefield = np.load(os.path.join(frq_wavefield_dir,file))
                wavefield /= np.max(np.abs(wavefield))
                real_frq_wavefield.append(wavefield.copy())
                
                # wavefield.close()
                del wavefield
                # for i in range(0,250,50):
                #     plt.imshow(real_frq_wavefield[-1][i,:,:])
                #     plt.title('Real')
                #     plt.show()

    if 'imag' in return_parameters:
        imag_frq_wavefield = []
        for file in file_list:
            if "imag_frqwavefield" in file:
                print(file)
                wavefield = np.load(os.path.join(frq_wavefield_dir,file))
                wavefield /= np.max(np.abs(wavefield))
                imag_frq_wavefield.append(wavefield.copy())
                
                # wavefield.close()
                del wavefield
                # for i in range(0,250,50):
                #     plt.imshow(imag_frq_wavefield[-1][i,:,:])
                #     plt.title('Imag')
                #     plt.show()

    # for i in range(len(real_frq_wavefield)):
    #     plt.figure(1)
    #     plt.subplot(2,2,1)
    #     plt.imshow(real_frq_wavefield[i][0,:,:])
    #     plt.clim(-1,1)
    #     plt.xlabel('z direction')
    #     plt.ylabel('x direction')
    #     plt.title('real')
    #     plt.set_cmap('seismic')
    #     plt.gca().invert_yaxis()

    #     plt.subplot(2,2,2)
    #     plt.imshow(imag_frq_wavefield[i][0,:,:])
    #     plt.clim(-1,1)
    #     plt.xlabel('z direction')
    #     plt.ylabel('x direction')
    #     plt.title('imag')
    #     plt.gca().invert_yaxis()
    #     # plt.set_cmap('Reds')

    #     plt.show()
    #         # plt.savefig(os.path.join(illumination_dir,file[:-4]))

    res = {}
    if 'pos' in return_parameters:
        res['pos'] = pos

    if 'real' in return_parameters:
        res['real'] = real_frq_wavefield
        res['shape'] = np.shape(real_frq_wavefield[0])
    
    if 'imag' in return_parameters:
        res['imag'] = imag_frq_wavefield
        res['shape'] = np.shape(imag_frq_wavefield[0])

    return res

def main():

    '''1. load training dataset'''
    res = load_scene_image()
    free_space_hashmap = res['free_space_index']
    len_free_space = len(free_space_hashmap)

    #Import frequency-domain data
    if a.mode == 'train' or a.mode == 'export':
        data_set = load_frq_wavefield_fix_window('train_dataset',return_parameters=['real','imag','pos'])
        if a.mode == 'export':
            a.n_epochs = 1
    elif a.mode == 'test':
        a.n_epochs = 1 #no need to repeated training
        data_set = load_frq_wavefield_fix_window('test_dataset',return_parameters=['real','imag','pos'])

    #define the mapping from source location to free_space_hashmap
    src_to_free_space = [] #store the source position in 1D free space domain (not the whole model domain)
    for i in range(len(data_set['pos'])):
        src_to_free_space.append(free_space_hashmap.index(data_set['pos'][i]))

    # #check whether the mapping from source to free space is correct:
    # for i in range(len(src_to_free_space)):
    #     print(free_space_hashmap[src_to_free_space[i]],data_set['pos'][i])

    '''2. set CNN architecture'''
    # net = UnetGenerator(input_nc=nf, output_nc=1)
    net = Net(a.nf*2,len_free_space)

    #The path to store CNN weights
    PATH = os.path.join(a.CNNweights_dir,'Unet.pth')
    if a.mode == 'train' or a.mode == 'export' or a.mode == 'test':
        if os.path.exists(PATH):
            net.load_state_dict(torch.load(PATH))
            print('Successfully load saved net!')
        else:
            print('No saved net loaded!')

    # net = Net() #PyTorch tutorial 
    print(net) 

    '''3. set GPU for deep learning'''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #PyTorch tutorial

    # Assuming that we are on a CUDA machine, this should print a CUDA device:  #PyTorch tutorial
    net.to(device)

    '''4. Define a Loss function and optimizer'''

    # import torch.optim as optim #PyTorch tutorial

    criterion = nn.CrossEntropyLoss() #PyTorch tutorial
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #PyTorch tutorial

    # criterion = nn.L1Loss() 
    optimizer = optim.Adam(net.parameters(), lr=a.lr, betas=(a.beta1, 0.999)) 

    '''4. Train the network'''

    loss_curve = [] # a.n_epochs*len(data_set['pos'])*a.ncenters
    loss_curve2 = []
    #Define the random robot locations for each source location
    src_list = [i for i in range(len(data_set['pos']))]

    count = 0

    if a.mode != 'train':
        src_real_list = []
        src_fake_list = []

    t_start = time.time()
    for epoch in range(a.n_epochs):  # iterative training
        #shuffle the source and robot locations simultaneously
        random.shuffle(src_list) #file index, [source position], [robot position]
        # print(src_list)
        # print('\n==================\n')
        for src_index in src_list: #loop over the different wavefield images
            count += 1

            #input wavefield as well as source location as CNN input and labels, respectively
            real_frqwvfd = data_set['real'][src_index]
            imag_frqwvfd = data_set['imag'][src_index]
            # plt.plot(real_frqwvfd[:,3,3])
            # plt.title(str(src_index))
            # plt.show()
            # imag_frqwvfd = data_set['imag'][file_index]

            

            input_data = np.zeros((1,a.nf*2,a.one_side*2+1,a.one_side*2+1)) #5-by-5
            label_data = np.zeros((1,len_free_space)) #only count space position

            #Define masked input
            #hardcode here: model is 500x500 but input data size is 512x512
            #so there is 6 points on each edge
            input_data[0,0:a.nf,:,:] = real_frqwvfd
            input_data[0,a.nf:a.nf*2,:,:] = imag_frqwvfd

            src_pos = src_to_free_space[src_index]
            label_data[0,src_pos] = 1
            

            # get the inputs; data is a list of [inputs, labels]
            # # for cpu:
            # inputs, labels = data
            # # for gpu:
            inputs = torch.FloatTensor(input_data)
            labels = torch.FloatTensor(label_data)
            inputs, labels = inputs.to(device), labels.to(device)
            # if inputs.is_cuda and labels.is_cuda:
            #      print("GPU will be used for PyTorch!")

            if a.mode == 'train':
                # zero the parameter gradients before running the backward pass.
                optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            # print('\n===============')
            # print(outputs[0,:,6+src_pos[0]-a.one_side:6+src_pos[0]+a.one_side+1,6+src_pos[1]-a.one_side:6+src_pos[1]+a.one_side+1])
            # print(labels[0,:,6+src_pos[0]-a.one_side:6+src_pos[0]+a.one_side+1,6+src_pos[1]-a.one_side:6+src_pos[1]+a.one_side+1])
            # # print(labels[0,0,6+src_pos[0],6+src_pos[1]])

            
            loss = criterion(outputs, labels)

            if a.mode == 'train':
                loss.backward()
                optimizer.step()

            loss_curve2.append(loss.item()*1000)
            loss_curve.append(torch.argmax(outputs)-torch.argmax(labels))

            if a.mode != 'train' or count % a.display <= 10:
                print('=========' + str(count) + '===============')
                t_end = time.time()
                print('time lapsed = ',t_end-t_start,'average time per src=',(t_end-t_start)/count)
                src_real = free_space_hashmap[torch.argmax(labels)]
                src_fake = free_space_hashmap[torch.argmax(outputs)]
                src_real_list.extend(src_real)
                src_fake_list.extend(src_fake)

                print('real source = ({},{})'.format(src_real[0],src_real[1]))
                print('estimated source = ({},{})'.format(src_fake[0],src_fake[1]))
                print(torch.sum(outputs[0,torch.argmax(outputs)-12:torch.argmax(outputs)+13]),
                      torch.sum(outputs[0,torch.argmax(labels)-12:torch.argmax(labels)+13]),
                      torch.sum(labels[0,torch.argmax(labels)-12:torch.argmax(labels)+13])
                )
                # print(outputs[0,torch.argmax(outputs)-12:torch.argmax(outputs)+13])
                # print(outputs[0,torch.argmax(labels)-12:torch.argmax(labels)+13])
                # print(labels[0,torch.argmax(labels)-12:torch.argmax(labels)+13])
                print('argmax(outputs)={},argmax(labels)={},max(outputs)={},max(labels)={}'.format(
                    torch.argmax(outputs),torch.argmax(labels),torch.max(outputs),torch.max(labels)))
                print('epoch={},count={},loss1={},loss2={}'.format(
                    epoch,count,loss_curve[-1],loss_curve2[-1]))
                print('=========' + str(count) + '===============')


        if a.mode == 'train':
            print('***************' + str(epoch) + '***************')
            t_end = time.time()
            print('time lapsed = ',t_end-t_start,'average time per src=',(t_end-t_start)/count)
            src_real = free_space_hashmap[torch.argmax(labels)]
            src_fake = free_space_hashmap[torch.argmax(outputs)]
            print('real source = ({},{})'.format(src_real[0],src_real[1]))
            print('estimated source = ({},{})'.format(src_fake[0],src_fake[1]))
            print(torch.sum(outputs[0,torch.argmax(outputs)-12:torch.argmax(outputs)+13]),
                    torch.sum(outputs[0,torch.argmax(labels)-12:torch.argmax(labels)+13]),
                    torch.sum(labels[0,torch.argmax(labels)-12:torch.argmax(labels)+13])
            )
            # print(outputs[0,torch.argmax(outputs)-12:torch.argmax(outputs)+13])
            # print(outputs[0,torch.argmax(labels)-12:torch.argmax(labels)+13])
            # print(labels[0,torch.argmax(labels)-12:torch.argmax(labels)+13])
            print('argmax(outputs)={},argmax(labels)={},max(outputs)={},max(labels)={}'.format(
                torch.argmax(outputs),torch.argmax(labels),torch.max(outputs),torch.max(labels)))
            print('epoch={},count={},loss1={},loss2={}'.format(
                epoch,count,loss_curve[-1],loss_curve2[-1]))
            print('***************' + str(epoch) + '***************')

    
    
        # plt.figure(1)
        # plt.subplot(2,2,1)
        # plt.imshow(input_data[0,0,:,:])
        # plt.clim(-1,1)
        # plt.subplot(2,2,2)
        # plt.imshow(label_data[0,0,:,:])
        # plt.clim(0,1)
        # plt.subplot(2,2,3)
        # plt.imshow(outputs[0,0,:,:])
        # plt.clim(0,1)
        # plt.subplot(2,2,4)
        # plt.imshow(outputs[0,0,:,:]-label_data[0,0,:,:])
        # plt.clim(-1,1)
        # plt.show()

    print('Finished Training') 
    if a.mode == 'train':
        print('save model after training')
        torch.save(net.state_dict(), PATH)
    else:
        np.savetxt(a.mode + '_src_real_list.txt',src_real_list, delimiter=',')  
        np.savetxt(a.mode + '_src_fake_list.txt',src_fake_list, delimiter=',')  

    print('save loss')
    torch.save(loss_curve,a.mode + '_loss.txt')
    torch.save(loss_curve2,a.mode + '_loss2.txt')


main()




