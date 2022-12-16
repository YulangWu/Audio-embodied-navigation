# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt
import math

import os
import random
from pysit import *
from pysit.gallery import horizontal_reflector

# from pysit.util.parallel import *

# from mpi4py import MPI

#from temporal_least_squares import TemporalLeastSquares # Integral method
import os
import argparse
from scipy.signal import hilbert

from BornModeling import BornModeling
from pysit.modeling.temporal_modeling import TemporalModeling

#   New Defined (Y.W.):
import copy

import numpy as np

from pysit.core.shot import *

from pysit.core.receivers import *
from pysit.core.sources import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def ricker(f, length=0.128, dt=0.001,amp = 10000):
    #f is peak frequncy
    #t = np.arange(-length/2, (length-dt)/2, dt)
    t = np.arange(0, length, dt)
    y = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))*amp
    # plt.plot(t,y)
    # plt.show()
    return t, y


def equispaced_acquisition(mesh, wavelet,
                           sources=1,
                           receivers='max',
                           source_depth=None,
                           source_kwargs={},
                           receiver_depth=None,
                           receiver_kwargs={},
                           list_of_source_location=[],
                           ):

    m = mesh
    d = mesh.domain

    xmin = d.x.lbound
    xmax = d.x.rbound

    zmin = d.z.lbound
    zmax = d.z.rbound

    if m.dim == 3:
        ymin = d.y.lbound
        ymax = d.y.rbound


    if source_depth is None:
        source_depth = zmin

    if receiver_depth is None:
        receiver_depth = zmin

    shots = list()

    max_sources = m.x.n

    if m.dim == 2:
        if receivers == 'max':
            receivers = m.x.n
        if sources == 'max':
            sources = m.x.n

        # if receivers > m.x.n:
        #     raise ValueError('Number of receivers exceeds mesh nodes.')
        # if sources > m.x.n:
        #     raise ValueError('Number of sources exceeds mesh nodes.')

        xpos = np.linspace(xmin, xmax, receivers)
        receiversbase = ReceiverSet(m, [PointReceiver(m, (x, receiver_depth), **receiver_kwargs) for x in xpos])

    for srcpos in list_of_source_location:
        # print('source is located at: ', srcpos)
        # Define source location and type
        source = PointSource(m, srcpos, wavelet, **source_kwargs)

        # Define set of receivers
        receivers = copy.deepcopy(receiversbase)

        # Create and store the shot
        shot = Shot(source, receivers)
        shots.append(shot)

    return shots


def peak_amp(signal):
    return np.max(np.abs(hilbert(signal)))

def gen_source_illumination(base_model, wavelet, src_pos, wall_lines_list, model_cpml_size, nt, nx, nz):
    forward_illumination = born_modeling.forward_model_with_2D_source(wavelet, src_pos, nx, nz, base_model,wall_lines_list, model_cpml_size, return_parameters=['illumination'])
    forward_illumination = np.array(forward_illumination['illumination'])
    forward_illumination = np.reshape(forward_illumination, (-1,nx*nz))
    forward_illumination.shape = nx, nz
    return forward_illumination

def gen_source_real_frqwavefield_illumination(base_model, wavelet, src_pos, wall_lines_list, model_cpml_size, nt, nx, nz, dt, frq_list,floor_plan):
    res = born_modeling.forward_model_with_2D_source(wavelet, src_pos, nx, nz, base_model,wall_lines_list, 
                        model_cpml_size, 
                        return_parameters=['wavefield','illumination'])

    forward_wavefield = np.array(res['wavefield'])
    
    # forward_illumination = np.array(res['illumination'])
    # print(np.shape(forward_wavefield),np.shape(forward_illumination))
    fn = 1/(nt*dt)
    nf = len(frq_list)
    real_frqwavefield = np.zeros((nf,nx*nz))
    # imag_frqwavefield = np.zeros((nf,nx*nz))
    for i in range(nx*nz):
            temp = np.fft.fft(forward_wavefield[:,i,0])
            for j in range(len(frq_list)):
                ana_frq = int(frq_list[j]/fn)
                real_frqwavefield[j,i] = np.real(temp[ana_frq])
                # imag_frqwavefield[j,i] = np.imag(temp[ana_frq])

    # real_frqwavefield.shape = nf,nx,nz        

    # print(np.shape(real_frqwavefield))
    # plt.subplot(1,2,1)s
    # plt.plot(real_frqwavefield[:,src_pos[0]*nz+src_pos[1]])
    # plt.subplot(1,2,2)

    '''For ploting time-domain wavefield only'''
    # illumination_iterative = np.zeros((nx,nz))
    # forward_wavefield.shape = -1, nx, nz
    # for i in range(nt-100):
    #     illumination_iterative += np.multiply(forward_wavefield[i,:,:],forward_wavefield[i,:,:])
        
    #     if i % 100 == 0:
    #         plt.imshow(np.log10(np.multiply(illumination_iterative,floor_plan)+np.ones((nx,nz))*0.00001))
    #         plt.clim(-10, 10)
    #         plt.set_cmap('seismic')
    #         plt.savefig('Rs_int_output/illumination' + '_' + str(src_pos[0]) + '_' + str(src_pos[1]) + '_' + str(i))
    #         # plt.show()

    illumination_iterative = None
    forward_wavefield = None

    res = {}
    res['real_frqwavefield'] = real_frqwavefield
    # res['imag_frqwavefield'] = imag_frqwavefield
    # res['illumination'] = forward_illumination

    return res

def collect_receiver_data(source_wavefields,robot_center,one_side=2):
    nt, nx, nz = np.shape(source_wavefields)
    
    c = robot_center 
    r = one_side
    seis_data = np.zeros((nt,r*2+1,r*2+1)) #the recorder array #the recorder array
    seis_data= source_wavefields[:,c[0]-r:c[0]+r+1,c[1]-r:c[1]+r+1]
    seis_data = np.reshape(seis_data,(nt,(r*2+1)**2))
    return seis_data

def check_validity(pos,floor_plan,one_side):
    
    x = pos[0]
    z = pos[1]

    if np.sum(floor_plan[x-one_side:x+one_side+1,z-one_side:z+one_side+1]) == (one_side*2+1)*(one_side*2+1):
        return True
    else:
        return False


##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################        
# solvers\constant_density_acoustic\time\constant_density_acoustic_time_base.py: set max_C = 5.0!!!!
if __name__ == '__main__':
#     # Setup

#     comm = MPI.COMM_WORLD
# #   comm = MPI.COMM_SELF
#     size = comm.Get_size()
#     rank = comm.Get_rank()

    # pwrap = ParallelWrapShot(comm=MPI.COMM_WORLD)6

# =============================================================================
# 1. Define parameters
# -----------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shots", type=float, default=100, help="Number of shots in the floor plan")
    parser.add_argument("--ncenters", type=int, default=1, help="Number of robot centers in the floor plan")
    parser.add_argument("--amplifier", type=int, default=2, help="increase the unit length by amplifier")
    parser.add_argument("--dh", type=float, default=0.01, help="dh = m")
    parser.add_argument("--t", type=int, default=30, help="time = ms") #25
    parser.add_argument("--v_air", type=float, default=0.343, help="indoor air velocity")#
    parser.add_argument("--v_wall", type=float, default=4.2, help="indoor wall velocity")
    parser.add_argument("--v_outside", type=float, default=0.343, help="outside velocity (include outer wall)")
    parser.add_argument("--cfl", type=float, default=1/1.414, help="temperal increment.")
    parser.add_argument("--one_side", type=int, default=2, help="one_side * 2 + 1 = maximum square in robot")
    parser.add_argument("--f_peak", type=int, default=0.2, help="Peak frequency")
    parser.add_argument("--display", type=int, default=200, help="for live plotting")
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

    parser.add_argument("--izs", type=int, default=500, help="useless")
    parser.add_argument("--izr", type=int, default=126, help="useless")

    
    # export options
    a = parser.parse_args()
    a.output_dir = a.model_name + '_' +a.output_dir

    a.dh *= a.amplifier #since a.amplifier will compress floor plan by amplifier times

    frq_list = [i for i in range(1,300)] #frequency band from low frequency to high frequency

    if a.model_name == None:
        print('Do no provide input file')
        exit()
    

    print('========================================================')
    print('Notice that turtlebot has radius=0.176m, maximum length of square = 0.2489m')
    print('The maximum number of units on the turtlebot = {}'.format(int(0.2489/0.01/a.amplifier)))
    for k, v in a._get_kwargs():
        print(k, "=", v)
    print('========================================================')

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)


    print('input file: {}'.format(a.model_name))
    # =============================================================================
    # 1. read floor plan to generate model
    # -----------------------------------------------------------------------------

    # 1.1 read floor plan
    
    floor_plan_no_furnitures = mpimg.imread(os.path.join(a.scenes_dir,os.path.join(a.model_name,a.type1)))
    floor_plan_furniture = mpimg.imread(os.path.join(a.scenes_dir,os.path.join(a.model_name,a.type2)))
    floor_plan_no_inner_wall = mpimg.imread(os.path.join(a.scenes_dir,os.path.join(a.model_name,a.type3)))
    #1 for free space, 0 for wall

    nx = len(floor_plan_no_furnitures)
    nz = nx
    floor_plan_no_furnitures.shape = nx, nz
    floor_plan_furniture.shape = nx, nz
    floor_plan_no_inner_wall.shape = nx, nz, 4
    floor_plan_no_inner_wall = floor_plan_no_inner_wall[:,:,0]

    floor_plan_no_furnitures = np.transpose(floor_plan_no_furnitures)
    floor_plan_furniture = np.transpose(floor_plan_furniture)
    floor_plan_no_inner_wall = np.transpose(floor_plan_no_inner_wall)

    if a.figs == True:
        plt.figure(1)
        plt.subplot(1,3,1)
        plt.imshow(np.transpose(floor_plan_no_furnitures))
        plt.title('floor plan (empty')

        plt.subplot(1,3,2)
        plt.imshow(np.transpose(floor_plan_furniture))
        plt.title('floor plan (furniture)')

        plt.subplot(1,3,3)
        plt.imshow(np.transpose(floor_plan_no_inner_wall))
        plt.title('floor plan (no wall)')

    
    #1.2 convert to the smaller one (e.g., nx/a.amplifier ==> 0.01m to 0.01m*amplifier ) 
    nz,nx = int(nz/a.amplifier),int(nx/a.amplifier)
    for i in range(nx):
        for j in range(nz):
            floor_plan_no_furnitures[i,j] = floor_plan_no_furnitures[i*a.amplifier,j*a.amplifier]
            floor_plan_furniture[i,j] = floor_plan_furniture[i*a.amplifier,j*a.amplifier]
            floor_plan_no_inner_wall[i,j] = floor_plan_no_inner_wall[i*a.amplifier,j*a.amplifier]
    floor_plan_no_furnitures = floor_plan_no_furnitures[:nx,:nz]
    floor_plan_furniture = floor_plan_furniture[:nx,:nz]
    floor_plan_no_inner_wall = floor_plan_no_inner_wall[:nx,:nz]
    floor_plan_inner_wall_only = floor_plan_no_inner_wall - floor_plan_no_furnitures
    floor_plan_wall_lines_only = np.ones((nx,nz))

    for i in range(2,nx-1):
        for j in range(2,nz-1):
            if floor_plan_no_furnitures[i,j] == 0:
                if np.sum(floor_plan_no_furnitures[i-1:i+2,j-1:j+2]) > 0:
                    floor_plan_wall_lines_only[i,j] = 0
            
    
    if a.figs == True:
        plt.figure(2)
        plt.subplot(2,2,1)
        plt.imshow(np.transpose(floor_plan_no_furnitures))
        plt.title('floor plan (floor_plan_no_furnitures)')

        plt.subplot(2,2,2)
        plt.imshow(np.transpose(floor_plan_no_inner_wall))
        plt.title('floor plan (floor_plan_no_inner_wall)')

        plt.subplot(2,2,3)
        plt.imshow(np.transpose(floor_plan_inner_wall_only))
        plt.title('floor plan (floor_plan_inner_wall_only)')

        plt.subplot(2,2,4)
        plt.imshow(np.transpose(floor_plan_wall_lines_only))
        plt.title('floor plan (floor_plan_wall_lines_only)')

        plt.show()

    # 1.3 define velocity model
    vel_model = np.zeros((nx,nz))

    wall_lines_list = [(0,0)]*nx*nz
    model_cpml_size = (nx+int(a.cpml_len/a.dh)*2,nz+int(a.cpml_len/a.dh)*2)

    #create wavefield using empty room (no furniture)
    count = 0
    for i in range(nx):
        for j in range(nz):
            #define the velocity in indoor area
            if floor_plan_no_inner_wall[i,j] == 1: #free space
                vel_model[i,j] = a.v_air
            else: #define the velocity in outdoor area and outer wall
                if floor_plan_wall_lines_only[i,j] == 1:
                    vel_model[i,j] = a.v_outside
                else:
                    vel_model[i,j] = a.v_wall

            #fix the wavefield in the inner wall to be 0
            if floor_plan_inner_wall_only[i,j] == 1:
                wall_lines_list[count]=(int(a.cpml_len/a.dh)+i,int(a.cpml_len/a.dh)+j)
                count += 1
    
    if a.figs == True:
        plt.imshow(vel_model)
        plt.title('acoustic model')
        plt.show()

    wall_lines_list = wall_lines_list[:count]
    

    free_space = [(0,0)]*(nx*nz)
    count = 0
    #create wavefield using empty room (no furniture)
    for i in range(nx):
        for j in range(nz):
            if floor_plan_furniture[i,j] == 1: #free space
                #add this free space to the list:
                free_space[count] = (i,j)
                count += 1

    #truncate the list
    free_space = free_space[0:count]
    print('size of free space = ',count)

    vel_model.shape = nx*nz, 1

    #1.4 define source locations (array) using floor plan with furniture
    floor_plan_furniture.shape = nx, nz
    source_pos = [] #store the different index of target center

    #number of shots in each scene
    nshots = a.num_shots #int(a.num_shots_ratio*len(free_space)/((a.one_side*2+1)**2))
    print('number of shots in this scene is', nshots)

    for i in range(nshots):
        while True:
            j = random.randrange(0,len(free_space))
            # print(check_validity(free_space[j],floor_plan_furniture,a.one_side))
            if check_validity(free_space[j],floor_plan_furniture,a.one_side) and free_space[j] not in source_pos:
                # if use pysit.forward_modeling, we need *a.dh
                #source_pos.append((a.dh*free_space[j][0],a.dh*free_space[j][1]))
                # as we use born_modeling, we do not need *a.dh
                source_pos.append((free_space[j][0],free_space[j][1]))
                break

    # =============================================================================
    # 2. set up PySIT parameters
    s_zpos = a.dh*a.izs
    r_zpos = a.dh*a.izr
    pmlx = PML(a.cpml_len, a.cpml_amp)
    pmlz = PML(a.cpml_len, a.cpml_amp)
    x_config = (0, nx*a.dh, pmlx, pmlx)
    z_config = (0, nz*a.dh, pmlz, pmlz)
    d = RectangularDomain(x_config, z_config)
    m = CartesianMesh(d, nx, nz)
    C, C0, m, d = horizontal_reflector(m)

    #cfl <= 1 always!!!
    dt = a.dh*a.cfl/a.max_v

    fmax = a.v_air/a.dh/a.n 
    nsteps = int(a.t/dt)
    trange = (0.0,a.t)

    #define wavelet form
    _, wavelet = ricker(a.f_peak, length=a.t, dt=dt)

    print('source location list',source_pos)
    print(
        'cfl={}; n={}; vmin={}; dh={}; nx={}; x={};dt=a.dh*a.cfl/a.max_v={}; t={}; nsteps= int(a.t/dt)={}; fmax=a.v_air/a.dh/16={}; fpeak={}'.format(
        a.cfl, a.n, a.v_air, a.dh,nx,a.dh*nx, dt, a.t, nsteps, fmax, a.f_peak
    ))
    # input('wait')


    C = vel_model.copy()

    # #Notice that equispaced_acquisition has been changed
    # shots = equispaced_acquisition(m,
    #                             RickerWavelet(a.f_peak),
    #                             sources=a.num_shots,
    #                             source_depth=s_zpos,
    #                             source_kwargs={},
    #                             receivers='max',
    #                             receiver_depth=r_zpos,
    #                             receiver_kwargs={},
    #                             list_of_source_location=source_pos,
    #                             )


     


    if a.figs == True:
        vp = floor_plan_no_furnitures.copy()
        vp.shape = nx, nz
        plt.subplot(1,2,1)
        plt.imshow(np.transpose(vp))
        plt.title('floor plan')

        vp = vel_model.copy()
        vp.shape = nx, nz
        plt.subplot(1,2,2)
        plt.imshow(np.transpose(vp))
        plt.title('True model')
        plt.show()

    solver = ConstantDensityAcousticWave(m,
                                        formulation='scalar',
                                        model_parameters={'C': C},
                                        spatial_accuracy_order=a.order,
                                        cfl_safety=a.cfl,
                                        trange=trange,
                                        kernel_implementation='cpp')

    born_modeling = BornModeling(solver)        #2D source data (nt,nx)
    # =============================================================================
    # 3. Generate observed data without direct wave
    # -----------------------------------------------------------------------------
    #free space now is 0 and wall is 1
    floor_plan_no_furnitures_reverse = np.ones((nx,nz)) - floor_plan_no_furnitures

    # Generate synthetic Seismic data
    print('Generating observed data...')
    base_model = solver.ModelParameters(m,{'C': C})
    
    for num_shot in range(len(source_pos)):
        source_filename = os.path.join(a.output_dir,
        os.path.splitext(a.model_name)[0] + '_saved_wavefield' + str(int(source_pos[num_shot][0])) + '_' + str(int(source_pos[num_shot][1])))
        tt1 = time.time()
        if os.path.exists(source_filename):
            print('The scene for this shot exists, so go to next shot')
            continue 
        else:
            print(source_filename)

        # 4.2.1.1 get source wavefield
        forward_illumination = None
        real_frqwavefield = None
        imag_frqwavefield = None

        res = gen_source_real_frqwavefield_illumination(base_model, wavelet, source_pos[num_shot], wall_lines_list, 
        model_cpml_size, nsteps, nx, nz, dt, frq_list,floor_plan_no_inner_wall)

        real_frqwavefield = res['real_frqwavefield']
        # imag_frqwavefield = res['imag_frqwavefield']
        # forward_illumination = res['illumination']

        tt2 = time.time()
        print('time=',tt2-tt1)
        real_frqwavefield.shape = len(frq_list), nx, nz
        # imag_frqwavefield.shape = len(frq_list), nx, nz
        # forward_illumination.shape = nx, nz

        np.save(
            os.path.join(a.output_dir,
            os.path.splitext(
            a.model_name)[0] + '_real_frqwavefield' + str(int(source_pos[num_shot][0])) + '_' + str(int(source_pos[num_shot][1]))), 
            real_frqwavefield)

        # np.save(
        #     os.path.join(a.output_dir,
        #     os.path.splitext(
        #     a.model_name)[0] + '_imag_frqwavefield' + str(int(source_pos[num_shot][0])) + '_' + str(int(source_pos[num_shot][1]))), 
        #     imag_frqwavefield)

        # np.save(
        #     os.path.join(a.output_dir,
        #     os.path.splitext(
        #     a.model_name)[0] + '_forward_illumination' + str(int(source_pos[num_shot][0])) + '_' + str(int(source_pos[num_shot][1]))), 
        #     forward_illumination)
 
        # saved_wavefield = np.zeros((nt,len(free_space)))    
        # for i in range(len(free_space)):           
        #     saved_wavefield[:,i] = wavefield[:,free_space[i][0],free_space[i][1]] 
        




























        # robot_center = [] #store the different index of robot center
        # for i in range(a.ncenters):
        #     while True:
        #         j = random.randrange(0,len(free_space))
        #         # print(check_validity(free_space[j],floor_plan_furniture,a.one_side))
        #         if check_validity(free_space[j],floor_plan_furniture,a.one_side) and free_space[j] not in source_pos:
        #             robot_center.append((free_space[j][0],free_space[j][1]))
        #             break


        






















        # print(robot_center)
        # seis_data_set = []
        # seis_data_set.append(robot_center)
        # for center in robot_center:
        #     seis_data = collect_receiver_data(wavefield,center,a.one_side)
        #     seis_data_set.append(seis_data)

            # if a.figs == True:
            # plt.figure(3)
            # plt.imshow(seis_data)
            # plt.clim(-0.01,0.01)
            # plt.title('recorded data')
            # plt.show()

        # np.save(source_filename, saved_wavefield)








       
        # # migration part:
        # for center in robot_center:
        #     seis_data = collect_receiver_data(wavefield,center,a.one_side)
        #     seis_data_set.append(seis_data)

            
        #     receiver_wavefields = None
        #     receiver_wavefields = gen_receiver_wavefield(base_model,wavefield,center,a.one_side, wall_lines_list)

        #     # receiver illumination and RTM image
        #     receiver_illumination = np.zeros((nx,nz))
        #     rtm_image = np.zeros((nx,nz))

        #     for i in range(nt):
        #         receiver_illumination += np.multiply(receiver_wavefields[i,:,:],receiver_wavefields[i,:,:])

        #         # #receiver_illumination snapshots
        #         # if a.figs == True and i % a.display == 0:
        #         #     x = np.multiply(floor_plan_no_furnitures,receiver_illumination)
            
        #     for i in range(nt):
        #         rtm_image += np.multiply(wavefield[i,:,:],receiver_wavefields[nt-1-i,:,:])

        #     np.save(os.path.join(a.output_dir,
        #     os.path.splitext(a.model_name)[0] + '_receiver_illumination' + 
        #     str(int(source_pos[num_shot][0])) + '_' + str(int(source_pos[num_shot][1]))) +
        #          '_' + str(center[0]) + '_' + str(center[1]), forward_illumination)

        #     np.save(os.path.join(a.output_dir,
        #     os.path.splitext(a.model_name)[0] + '_RTM' + str(int(source_pos[num_shot][0])) + '_' + 
        #     str(int(source_pos[num_shot][1]))) \
        #          + '_' + str(center[0]) + '_' + str(center[1]), rtm_image)
                 
        #     forward_illumination = np.multiply(floor_plan_no_furnitures,forward_illumination)
        #     receiver_illumination = np.multiply(floor_plan_no_furnitures,receiver_illumination)
        #     rtm_image = np.multiply(floor_plan_no_furnitures,rtm_image)

            

        #     # if a.figs == True:
        #     plt.figure(1)
        #     plt.subplot(1,3,1)
        #     plt.imshow(np.transpose(forward_illumination))
        #     plt.title('Source illumination')

        #     plt.subplot(1,3,2)
        #     plt.imshow(np.transpose(receiver_illumination))
        #     plt.title('Receiver illumination')
        #     plt.subplot(1,3,3)
        #     plt.imshow(np.transpose(rtm_image))
        #     plt.title('RTM image')
        #     plt.show()

        #     # plt.savefig(os.path.join(a.output_dir,
        #     # os.path.splitext(a.model_name)[0] + '_RTM' + str(int(source_pos[num_shot][0])) + 
        #     # '_' + str(int(source_pos[num_shot][1]))) \
        #     #      + '_' + str(center[0]) + '_' + str(center[1]))

