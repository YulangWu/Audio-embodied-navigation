

import numpy as np
from pysit.util.derivatives import build_derivative_matrix, build_permutation_matrix, build_heterogenous_matrices
from numpy.random import uniform
import time
import matplotlib.pyplot as plt
__all__ = ['BornModeling']

__docformat__ = "restructuredtext en"


class BornModeling(object):
    """Class containing a collection of methods needed for seismic inversion in
    the time domain.

    This collection is designed so that a collection of like-methods can be
    passed to an optimization routine, changing how we compute each part, eg, in
    time, frequency, or the Laplace domain, without having to reimplement the
    optimization routines.

    A collection of inversion functions must contain a procedure for computing:
    * the foward model: apply script_F (in our notation)
    * migrate: apply F* (in our notation)
    * demigrate: apply F (in our notation)
    * Hessian?

    Attributes
    ----------
    solver : pysit wave solver object
        A wave solver that inherits from pysit.solvers.WaveSolverBase

    """

    # read only class description
    @property
    def solver_type(self): return "time"

    @property
    def modeling_type(self): return "time"

    def __init__(self, solver):
        """Constructor for the TemporalInversion class.

        Parameters
        ----------
        solver : pysit wave solver object
            A wave solver that inherits from pysit.solvers.WaveSolverBase

        """

        if self.solver_type == solver.supports['equation_dynamics']:
            self.solver = solver
        else:
            raise TypeError("Argument 'solver' type {1} does not match modeling solver type {0}.".format(
                self.solver_type, solver.supports['equation_dynamics']))

    def virtual_source(self, operand_simdata, perturbation_model,dt):
        #### REMEMBER COPY!!! OTHERWISE, ORIGINAL WAVEFIELD IS ERASED!!!
        dt2 = dt*dt
        nt = len(operand_simdata[:,0])
        totaln = len(operand_simdata[0,:])
        # print('size of data is {}x{}'.format(nt,totaln))

        # 1. second-order time-derivatives of background wavefield (operand_simdata)
        temp_wavefield = np.zeros((2,len(operand_simdata[0,:]))) # dim = [2, nx*nz]
        temp_wavefield[0,:] = operand_simdata[0,:]
        temp_wavefield[1,:] = operand_simdata[1,:]
        

        for t in range(1, nt - 1):
            operand_simdata[t,:] = temp_wavefield[0,:] - 2*temp_wavefield[1,:] + operand_simdata[t+1,:]
            temp_wavefield[0,:] = temp_wavefield[1,:]
            temp_wavefield[1,:] = operand_simdata[t+1,:]

        operand_simdata[0,:] = 0
        
        # operand_simdata = np.diff(operand_simdata, n = 2, axis = 0)
        # 2. get virtual source
        for i in range(totaln):
            operand_simdata[:,i] = -np.multiply(operand_simdata[:,i],perturbation_model[i])

        return operand_simdata/dt2 

    def _setup_forward_rhs_with_2D_source(self, rhs_array, k, input_source):
        # basic rhs is always the pseudodata or residual
        #input 3D input source is too expensive!!! waste memory!
        # rhs_array = self.solver.mesh.pad_array(input_source[k,:] out_array=rhs_array)
        rhs_array = self.solver.mesh.pad_array(input_source, out_array=rhs_array)
        return rhs_array       

    def forward_model_with_2D_source(self, wavelet, src_pos, nx, nz, m0, fix_val_list, model_cpml_size, return_parameters=[]):
        """Applies the forward model to the model for the given solver.

        Parameters
        ----------
        shot : pysit.Shot
            Gives the source signal approximation for the right hand side.
        m0 : solver.ModelParameters
            The parameters upon which to evaluate the forward model.
        return_parameters : list of {'wavefield', 'simdata', 'dWaveOp'}

        Returns
        -------
        retval : dict
            Dictionary whose keys are return_parameters that contains the specified data.

        Notes
        -----
        * u is used as the target field universally.  It could be velocity potential, it could be displacement, it could be pressure.
        * utt is used to generically refer to the derivative of u that is needed to compute the imaging condition.

        Forward Model solves:

        For constant density: m*u_tt - lap u = f, where m = 1.0/c**2
        For variable density: m1*u_tt - div(m2 grad)u = f, where m1=1.0/kappa, m2=1.0/rho, and C = (kappa/rho)**0.5
        """

        # Local references
        solver = self.solver
        solver.model_parameters = m0

        mesh = solver.mesh

        d = solver.domain
        dt = solver.dt
        input_source = np.zeros((nx*nz,1))
        nsteps = len(wavelet)
        model_size =nx*nz
        
        # print('nsteps=',nsteps)
        # Storage for the field
        if 'wavefield' in return_parameters:
            us = list()

        # Storage for the time derivatives of p
        if 'dWaveOp' in return_parameters:
            dWaveOp = list()

        if 'illumination' in return_parameters:
            illumination = np.zeros((model_size,1))

        # Step k = 0
        # p_0 is a zero array because if we assume the input signal is causal
        # and we assume that the initial system (i.e., p_(-2) and p_(-1)) is
        # uniformly zero, then the leapfrog scheme would compute that p_0 = 0 as
        # well. ukm1 is needed to compute the temporal derivative.
        solver_data = solver.SolverData()

        rhs_k = np.zeros(mesh.shape(include_bc=True))
        rhs_kp1 = np.zeros(mesh.shape(include_bc=True))

        tt1 = time.time()
        for k in range(nsteps-1):
            if k % 1000 == 0:
                tt2 = time.time()
                t_per_inter = (tt2-tt1)/(k+1)

                print('total step=',nsteps,'step=',k, 
                'total time= ', t_per_inter*nsteps, 'time left=',t_per_inter*(nsteps-1-k))

            uk = solver_data.k.primary_wavefield
            uk_bulk = mesh.unpad_array(uk)

            #fix the wavefields in the inner wall to 0
            for index in fix_val_list:
                #as np.transpose(solver_data.k.primary_wavefield) = (nx, nz), so nz is the horizontal line,
                #so the one-dimensional data is stored line by line, namely, from 0 to nz-1 for x=0,1,2,nx
                solver_data.k.primary_wavefield[index[0]*model_cpml_size[1] + index[1]] = 0

            if 'wavefield' in return_parameters:
                us.append(uk_bulk.copy())


            if 'illumination' in return_parameters:
                u_temp = uk_bulk.copy()
               
                illumination += np.multiply(u_temp,u_temp)


            if k == 0:
                input_source[src_pos[0]*nz+src_pos[1],0] = wavelet[k]
                rhs_k = self._setup_forward_rhs_with_2D_source(
                    rhs_k, k,   input_source)
                
                input_source[src_pos[0]*nz+src_pos[1],0] = wavelet[k+1]
                rhs_kp1 = self._setup_forward_rhs_with_2D_source(
                    rhs_kp1,  k+1, input_source)
            else:
                input_source[src_pos[0]*nz+src_pos[1],0] = wavelet[k+1]
                # shift time forward
                rhs_k, rhs_kp1 = rhs_kp1, rhs_k
            rhs_kp1 = self._setup_forward_rhs_with_2D_source(
                rhs_kp1, k+1, input_source)

            

            # Note, we compute result for k+1 even when k == nsteps-1.  We need
            # it for the time derivative at k=nsteps-1.
            solver.time_step(solver_data, rhs_k, rhs_kp1)
            

            # When k is the nth step, the next step is uneeded, so don't swap
            # any values.  This way, uk at the end is always the final step
            if(k == nsteps - 1):
                break

            # Don't know what data is needed for the solver, so the solver data
            # handles advancing everything forward by one time step.
            # k-1 <-- k, k <-- k+1, etc
            solver_data.advance()

        if 'wavefield' in return_parameters:
            us.append(np.empty_like(us[-1]))


        retval = dict()

        if 'wavefield' in return_parameters:
            retval['wavefield'] = us

        if 'illumination' in return_parameters:
            retval['illumination'] = illumination

        return retval

    def forward_model_with_2D_BC(self, input_data, m0, depth, nx, nz, n_pml, return_parameters=[]):
        """Applies the forward model to the model for the given solver.

        Parameters
        ----------
        shot : pysit.Shot
            Gives the source signal approximation for the right hand side.
        m0 : solver.ModelParameters
            The parameters upon which to evaluate the forward model.
        return_parameters : list of {'wavefield', 'simdata', 'dWaveOp'}

        Returns
        -------
        retval : dict
            Dictionary whose keys are return_parameters that contains the specified data.

        Notes
        -----
        * u is used as the target field universally.  It could be velocity potential, it could be displacement, it could be pressure.
        * utt is used to generically refer to the derivative of u that is needed to compute the imaging condition.

        Forward Model solves:

        For constant density: m*u_tt - lap u = f, where m = 1.0/c**2
        For variable density: m1*u_tt - div(m2 grad)u = f, where m1=1.0/kappa, m2=1.0/rho, and C = (kappa/rho)**0.5
        """

        # Local references
        solver = self.solver
        solver.model_parameters = m0

        mesh = solver.mesh

        d = solver.domain
        dt = solver.dt
        nsteps = len(input_data)
        # print('nsteps=',nsteps)
        # Storage for the field
        if 'wavefield' in return_parameters:
            us = list()

        # Storage for the time derivatives of p
        if 'dWaveOp' in return_parameters:
            dWaveOp = list()

        # Step k = 0
        # p_0 is a zero array because if we assume the input signal is causal
        # and we assume that the initial system (i.e., p_(-2) and p_(-1)) is
        # uniformly zero, then the leapfrog scheme would compute that p_0 = 0 as
        # well. ukm1 is needed to compute the temporal derivative.
        solver_data = solver.SolverData()

        rhs_k = np.zeros(mesh.shape(include_bc=True))
        rhs_kp1 = np.zeros(mesh.shape(include_bc=True))

        for k in range(nsteps-1):  # xrange(int(solver.nsteps)):

            # 1.First get unpad array to insert data as boundary condition:
            orig_u = solver_data.k.primary_wavefield.copy()
            temp_u = mesh.unpad_array(orig_u)
            temp_u.shape = nx, nz
            orig_u.shape = nx+2*n_pml,nz+2*n_pml

            # 2.Then insert data as boundary condition at correct depth in unpad array:
            temp_u[:,depth] = input_data[k,:]

            # 3.Copy unpad array to the copy of the original pad array orig_u
            # notice that only copy of solver_data.k.primary_wavefield can be
            # reshaped, so this copy orig_u is very important and necessary!
            orig_u[n_pml:n_pml+nx,n_pml:n_pml+nz] = temp_u

            # if k % 100 == 0 :
            #     plt.imshow(orig_u,aspect='auto')
            #     plt.set_cmap('seismic')
            #     plt.clim(-0.001,0.001)
            #     plt.show()
            
            # 4.Assign solver_data.k.primary_wavefield the orig_u (both have same shape)
            orig_u.shape = -1, 1
            solver_data.k.primary_wavefield = orig_u 

            uk = solver_data.k.primary_wavefield
            uk_bulk = mesh.unpad_array(uk)

            if 'wavefield' in return_parameters:
                us.append(uk_bulk.copy())

            # Note, we compute result for k+1 even when k == nsteps-1.  We need
            # it for the time derivative at k=nsteps-1.
            solver.time_step(solver_data, rhs_k, rhs_kp1)


            # When k is the nth step, the next step is uneeded, so don't swap
            # any values.  This way, uk at the end is always the final step
            if(k == nsteps - 1):
                break

            # Don't know what data is needed for the solver, so the solver data
            # handles advancing everything forward by one time step.
            # k-1 <-- k, k <-- k+1, etc
            solver_data.advance()

        if 'wavefield' in return_parameters:
            us.append(np.empty_like(us[-1]))

        retval = dict()

        if 'wavefield' in return_parameters:
            retval['wavefield'] = us
        return retval