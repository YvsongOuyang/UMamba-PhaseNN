from scipy.ndimage import gaussian_filter
import numpy as np
import pylab as plt
import os
from pynx.scattering.fhkl import Fhkl_thread
from pynx.scattering.fthomson import f_thomson
import random
from numpy.fft import fftshift, ifftshift, fftn, ifftn
from Global_utilities import *
from Object_utilities import *
from Plot_utilities import *
from PostProcessing import *
import itertools

######################################################################################################################################
################################           Read lmp file of the particle             ##########################################################
######################################################################################################################################


def read_lmp(filepath):
    '''
    Read the atomic positions X,Y,Z from a .lmp file
    '''
    Natoms = int(np.loadtxt(filepath, skiprows = 2, usecols = 0, max_rows = 2)[0])
    
    file = open(filepath)
    n_line = 0
    line = file.readline()
    while 'Atoms # atomic' not in line:
        line = file.readline()
        n_line += 1
        
    X,Y,Z = np.loadtxt(filepath,skiprows=n_line+1,usecols = (2,3,4),unpack=True, max_rows = Natoms)
    
    return X,Y,Z

######################################################################################################################################
################################           Random rotation in Q space             ##########################################################
######################################################################################################################################


def RandomdqsRotation(sizeQ,dqx,dqy,dqz):
    '''
    Make a random rotation of the dq's
    '''
    R = rand_rotation_matrix()
    dqxqyqz = np.vstack([dqx.ravel(), dqy.ravel(), dqz.ravel()]).T
    dqxqyqz1 = np.array([np.dot(R,i) for i in dqxqyqz])

    dqx = dqxqyqz1[:,0]
    dqy = dqxqyqz1[:,1]
    dqz = dqxqyqz1[:,2]

    dqx = dqx.reshape((sizeQ,sizeQ,sizeQ))
    dqy = dqy.reshape((sizeQ,sizeQ,sizeQ))
    dqz = dqz.reshape((sizeQ,sizeQ,sizeQ))
    return dqx,dqy,dqz

######################################################################################################################################
################################           Create Qx, Qy, Qz meshgrid             ##########################################################
######################################################################################################################################

def Createqxqyqz(dq,sizeQ,hkl,element,
                 random_rotation=True,
                 random_shift=True, max_pixel_random_shift=1.):
    '''
    Create my 3D grid in reciprocal space (the q's at which I will calculate the diffraction of the crystal)
    I create a dq grid,
    make a random rotation,
    make a random shift to avoid calculating exactly the center of the Bragg peak as it is not the case experimentally,
    finally I add hkl/lattice to the dq's to center my grid around the Bragg peak
    
    :sizeQ: Size of my 3D image
    :rotation: make a random rotation of the whole q space if True (equivalent to rotating the particle)
    :random_shift: make a small random shift to avoid calculating the exact center
    :max_pixel_random_shift: (float) the maximum possible shift in pixel. This can be changed but shouldn't be too large.
    '''
    lattice_dict = {'Au':4.080, 'Ag':4.090, 'Al':4.050, 'Pt': 3.9201}
    
    if element is None:
        element = 'Au'
        
    lattice = lattice_dict[element] # lattice parameter of the particle

    # Create my dq's grid
    dqx = (np.arange(sizeQ)-sizeQ/2)*dq # extent in the reciprocal space
    dqy = (np.arange(sizeQ)-sizeQ/2)*dq
    dqz = (np.arange(sizeQ)-sizeQ/2)*dq
    dqx,dqy,dqz = np.meshgrid(dqx,dqy,dqz)

    # Make a random rotation
    if random_rotation==True:
        dqx,dqy,dqz = RandomdqsRotation(sizeQ,dqx,dqy,dqz)

    # Make a random shift
    if random_shift==True:
        shiftx = dq*np.random.rand()*max_pixel_random_shift 
        shifty = dq*np.random.rand()*max_pixel_random_shift
        shiftz = dq*np.random.rand()*max_pixel_random_shift
        dqx = dqx+shiftx
        dqx = dqx+shifty
        dqx = dqx+shiftz
    
    # Make the grid around the Bragg peak hkl
    qx = hkl[0]/lattice+dqx
    qy = hkl[1]/lattice+dqy
    qz = hkl[2]/lattice+dqz
    return qx,qy,qz

######################################################################################################################################
################################           Create random rotation matrix in Q space           ##########################################################
######################################################################################################################################


def rand_rotation_matrix(deflection=1.0, randnums=None):    
    '''
    Make a random rotation matrix that I will use to rotate the reciprocal lattice q
    EB found that on a forum. I hope that the rotations probabilities are uniformly distributed.
    '''
    if randnums is None:
        randnums = np.random.uniform(size=(3,))       
    theta, phi, z = randnums 
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    st = np.sin(theta)
    ct = np.cos(theta)
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M
from scipy.ndimage.measurements import center_of_mass

######################################################################################################################################
################################           Re - center the Bragg peak           ##########################################################
######################################################################################################################################

def CenterTheCenterOfMass(Fexp):
    '''
    Center the center of mass of a 3D matrix 
    I use this to center the Bragg peak after the small random shift I did in the function "Createqxqyqz"
    '''
    nbz,nby,nbx = Fexp.shape
    Fexp[~np.isfinite(Fexp)] = 0
    intensity = abs(Fexp)**2
    
    # Calculate where is the center of mass
    piz,piy,pix = center_of_mass(intensity)
    
    # Calculate what's the offset to put back the center of mass at the middle of the 3D matrix
    offset_z = int(np.rint(nbz / 2.0 - piz))
    offset_y = int(np.rint(nby / 2.0 - piy))
    offset_x = int(np.rint(nbx / 2.0 - pix))

    # Put back the center of mass to the middle of the matrix
    Fexp = np.roll(Fexp, (offset_z, offset_y, offset_x), axis=(0, 1, 2))
    return Fexp

######################################################################################################################################
################################           Create diffraction Fexp           ##########################################################
######################################################################################################################################


def Create_diffraction(qx,qy,qz,
                       x,y,z,
                       hkl,
                       element,
                       center_the_center_of_mass=False,
                       return_diffracted_amplitude=True):
    '''
    Create the 3D diffraction array
    : qx, qy, qz : three 3D arrays with the corresponding reciprocal space wavevectors
    : x, y, z : three 1D arrays of the 3D atomic positions of the crystal
    : hkl : Bragg wavevector, needed for the "s" factor
    : element : string of the atomic element
    : center_the_center_of_mass : if True, center the 3D diffracted array around its center of mass
    : return_diffracted_amplitude : if True, return the complex diffracted amplitude instead of the diffracted intensity
    '''
    lattice_dict = {'Au':4.080, 'Ag':4.090, 'Al':4.050, 'Pt': 3.9201}
    lattice = lattice_dict[element] # lattice parameter of the particle
 
    fhkl, dt =Fhkl_thread(qx,qy,qz,x,y,z,occ=None, gpu_name="",language="cuda", verbose=False)
    s = (1/2)*(np.sqrt(hkl[0]**2+hkl[1]**2+hkl[2]**2)/lattice) # sin(theta)/lambda
    fEl = f_thomson(s, element)
    Fexp = fhkl*fEl
    
    if center_the_center_of_mass: 
        Fexp = CenterTheCenterOfMass(Fexp)
        
    I = np.abs(Fexp)**2 # I can go to float32 since that's what I'll use in tensorflow
    
    if return_diffracted_amplitude:
        return Fexp
    else:
        return I
    
#################################################################################################################################################################################################################################################################
#####################################################                                     FUNCTIONS ON THE OBJECT IN REAL SPACE                                      ############################################################################################
#################################################################################################################################################################################################################################################################

def smooth_object(obj, 
                  seed = False, 
                  sigma_gaussian=None,
                  plot=False):
    
    if seed:
        np.random.seed(42)
        
    if sigma_gaussian is None:
        sigma_gaussian = np.random.uniform(.45, .75) # CHOSEN BY EB. Don't hesitate to change
        
    module = np.abs(obj)
    module_smooth = gaussian_filter(module, sigma=sigma_gaussian)
    obj_smooth = module_smooth * np.exp(1.0j * np.angle(obj))
    
    if plot:
        plot_2D_slices_middle_only_module(obj, fig_title='object module')
        plot_2D_slices_middle_only_module(obj_smooth, fig_title='object module after smoothing')
    return obj_smooth 

######################################################################################################################################
################################            Clean the support             ##########################################################
######################################################################################################################################

def remove_real_space_module_out_support(obj,
                                         threshold_module=.1,  # CHOSEN BY EB. Don't hesitate to change
                                         plot=False):
    module = np.abs(obj)
    support = np.array(module > np.max(module)*threshold_module, dtype='int')
    module[support==0] = 0.
    
    obj_clean = module * np.exp(1.0j*np.angle(obj))
    
    if plot:
        plot_2D_slices_middle_only_module(support, fig_title='support using a threshold on the module')
        plot_2D_slices_middle_only_module(obj_clean, fig_title='module after removing everything ouside the support')
    return obj_clean

######################################################################################################################################
################################            Define random noise             ##########################################################
######################################################################################################################################

from scipy.signal import fftconvolve
def random_noise(size, 
                 seed = False,
                 correlation_length = None, 
                 noise_scale = 1.,
                 plot=False):
    
    if correlation_length is None:
        correlation_length = np.random.uniform(.01,.1)  # CHOSEN BY EB. Don't hesitate to change
        
    if seed:
        np.random.seed(42)
    
    f = np.random.normal(size = (size, size, size))
    x,y,z = np.meshgrid(np.linspace(-1,1,size), np.linspace(-1,1,size), np.linspace(-1,1,size))
    kernel = np.exp(-x**2./(2.*correlation_length**2.) -y**2./(2.*correlation_length**2.) - z**2./(2.*correlation_length**2.))
    
    noise = fftconvolve(kernel, f, mode='same')
    
    noise = noise * noise_scale/np.max(noise)
    
    if plot:
        plot_2D_slices_middle_one_array3D(noise)
    return noise


######################################################################################################################################
################################            Add random noise to reals space modulus           ##########################################
######################################################################################################################################

def add_random_noise_module(module,
                            seed = False, 
                            module_factor = None,
                            correlation_length=None,
                            plot=False):
    
    if module_factor is None:
        module_factor = np.random.uniform(.5,3.)  # CHOSEN BY EB. Don't hesitate to change
    
    module_range = module_factor*np.max(module)

    module_var = random_noise(module.shape[0], 
                             correlation_length = correlation_length, seed = seed)

    module_var = module_range*(module_var-np.min(module_var))/(np.max(module_var)-np.min(module_var))
    
    module_var = module_var * module/np.max(module) # Make sure to add only where the module is not 0
    
    if plot:
        plot_2D_slices_middle_only_module(module_var, fig_title='noise added to module')
        plot_2D_slices_middle_only_module(module+module_var, fig_title='module + noise')
    return module+module_var


######################################################################################################################################
################################            Add random real space phase with seed            ##########################################
######################################################################################################################################

def replace_phase_by_random_phase(obj,
                                  phase_range=None,
                                  correlation_length=None, 
                                  seed = False,
                                  plot=False):
    if seed: 
        np.random.seed(42)
        
    if phase_range is None:
        phase_range = np.random.uniform(1.5*pi, 5*pi)  # CHOSEN BY EB. Don't hesitate to change
        
    phase = random_noise(obj.shape[0], seed = seed, 
                             correlation_length = correlation_length)

    phase = phase_range*(phase-np.min(phase))/(np.max(phase)-np.min(phase))-phase_range/2.
    
    obj_new_phase = np.abs(obj) * np.exp(1.0j*phase)

    if plot:
        plot_2D_slices_middle(obj_new_phase, fig_title='object with replaced phase')
    return obj_new_phase 


##########################################################################################################################################################################
######################################################     Gaussian Strain     ###########################################################################################
##########################################################################################################################################################################


from numpy import pi
import random

def simulate_strain_gauss(obj,
                    phase_range1= None,
                    phase_range2 = None,
                    sigma1 = None, 
                    sigma2 = None,
                    plot=False):
    
    '''
    replace the phase of the object with two 2D Gaussians, one positive and one negative with random height and width
    
    '''

    if phase_range1 is None:
        
        phase_range1 = np.random.uniform(3*pi, 4*pi) # random values, play with it as you prefer
        
    if phase_range2 is None:
        
        phase_range2 = np.random.uniform(2.5*pi, 3.5*pi) # random values, play with it as you prefer
    
    x,y,z = np.indices(obj.shape)
    
    # evaluate the average size of the support in order to have the peaks of the Gaussians within the support
    # not very well coded but it does the job for the moment
    mod = np.abs(obj)
    mod[mod<.2*np.max(mod)] = 0 
    mod[mod>= .2*np.max(mod)] = 1
    vol = np.sum(mod)
    avg_side = int(((vol/(obj.shape[0]*obj.shape[1]*obj.shape[2]))**(1/3)) *obj.shape[0])
    
    # set the width of the Gaussian
    if sigma1 is None or sigma2 is None: 
        sigma1 = np.random.uniform(60,300)
        sigma2 = np.random.uniform(70,200)
    
    # centers of the Gaussians, close to the center of the array
    center1 = np.random.randint(obj.shape[0]//2 - obj.shape[0]//10,obj.shape[0]//2 + obj.shape[0]//10 )
    center2 = center1 + np.random.randint(-avg_side//2, +avg_side//2)
    
    phase_z = phase_range1*np.exp(-(((x-center1)**2)/sigma1 + ((y-center2)**2)/sigma2)) - phase_range2*np.exp(-(((x-center2)**2)/sigma2 + ((y-center1)**2)/sigma1))
    phase_x = phase_range1*np.exp(-(((y-center1)**2)/sigma1 + ((z-center2)**2)/sigma2)) - phase_range2*np.exp(-(((y-center2)**2)/sigma2 + ((z-center1)**2)/sigma1))
    phase_y = phase_range1*np.exp(-(((z-center1)**2)/sigma1 + ((x-center2)**2)/sigma2)) - phase_range2*np.exp(-(((z-center2)**2)/sigma2 + ((x-center1)**2)/sigma1))
    
    
    # choose randomly one of the three
    phase = random.choice([phase_z, phase_x, phase_y])
    
    obj_new_phase = np.abs(obj) * np.exp(1.0j*phase)
    
    # remove any possible phase ramp
    obj_new_phase  = remove_phase_ramp(obj_new_phase ,
                            threshold_module=.3,
                            crop=False,
                            return_ramp=False,
                            method='fit', # 'gradient'
                            plot=False) 

    if plot:
        plot_2D_slices_middle(obj_new_phase, fig_title='object with replaced phase')
    return obj_new_phase 

##########################################################################################################################################################################
######################################################     Cosine Strain     ###########################################################################################
##########################################################################################################################################################################


def simulate_strain_cosine(obj,
                    phase_range1= None,
                    phase_range2 = None,
                    sigma1 = None, 
                    sigma2 = None,
                    seed = False,
                    plot=False):
    '''
    replace the phase of the object with two 2D cosines.
    
    '''
    
    if phase_range1 is None:
        
        phase_range1 = np.random.uniform(3*pi, 4*pi) # random values, play with it as you prefer
        
    if phase_range2 is None:
        phase_range2 = np.random.uniform(2.5*pi, 3.5*pi) # random values, play with it as you prefer

    x,y,z = np.indices(obj.shape)
    
    mod = np.abs(obj)
    mod[mod<.2*np.max(mod)] = 0 
    mod[mod>= .2*np.max(mod)] = 1
    vol = np.sum(mod)
    avg_side = int(((vol/(obj.shape[0]*obj.shape[1]*obj.shape[2]))**(1/3)) *obj.shape[0])
    
    if sigma1 is None or sigma2 is None: 
        sigma1 = np.random.uniform(60,300)
        sigma2 = np.random.uniform(70,200)
    
    center1 = np.random.randint(obj.shape[0]//2 - obj.shape[0]//10,obj.shape[0]//2 + obj.shape[0]//10 )
    center2 = center1 + np.random.randint(-avg_side//2, +avg_side//2)
    
    a = np.random.uniform(.5, 2)/avg_side
    b = np.random.uniform(.5, 2)/avg_side
    c = np.random.uniform(.5, 2)/avg_side
    d = np.random.uniform(.5, 2)/avg_side

    phase_x = phase_range1*np.cos(a*x + b*y - center1) + phase_range2*np.cos(c*z + d*x - center2)
    phase_y = phase_range1*np.cos(a*y + b*z - center1) + phase_range2*np.cos(c*y + d*x - center2)
    phase_z = phase_range1*np.cos(a*z + c*x - center1) + phase_range2*np.cos(c*z + d*y - center2)
    
    phase = random.choice([phase_z, phase_x, phase_y])
    
    obj_new_phase = np.abs(obj) * np.exp(1.0j*phase)
    obj_new_phase = remove_phase_ramp(obj_new_phase,
                      threshold_module=.3,
                      crop=False,
                      return_ramp=False,
                      method='fit', # 'gradient'
                      plot=False) 
    if plot:
        plot_2D_slices_middle(obj_new_phase, fig_title='object with replaced phase')
        
    return obj_new_phase 

##########################################################################################################################################################################
######################################################     Quadratic Strain     ###########################################################################################
##########################################################################################################################################################################
# import scipy.stats
# from scipy.stats import uniform_direction
def simulate_strain_quad_substrate(obj,
                                   threshold_module=.3,
                                   phase_range = None, axis_quadratic = None, quad_along_axis=None,
                                   add_random_small_phase = True, small_phase_range=None, corr_phase=None,
                                   plot=False, verbose=False):
    
    '''
    replace the phase of the object with quadratic phase to simualte substrate induced deformation
    
    :add_random_small_phase: add a small random phase on top of the large quadratic one. 
    '''
    
    if phase_range is None:
        phase_range = np.random.uniform(pi, 4*pi) # random values, play with it as you prefer
        
    if axis_quadratic is None:
        if quad_along_axis is None:
            if np.random.randint(2):
                if verbose :
                    print("quadratic phase along one axis of your array")
                axis_quadratic = np.array([0,0,0])
                axis_quadratic[np.random.randint(3)] = 1
            else:
                 
                theta = np.random.uniform(0, 2 * np.pi)  # Random azimuthal angle
                phi = np.random.uniform(0, np.pi)        # Random polar angle
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                axis_quadratic = np.array([x, y, z]) 
                
                # axis_quadratic = scipy.stats.uniform_direction.rvs(3)
    
    pos = np.indices(obj.shape)
    com = np.array(center_of_mass(obj))
    pos = pos - com[:,None,None,None]
    
    phase_quad = np.sum(pos * axis_quadratic[:, None,None,None],axis=0) **2.
    
    # add a small random phase
    if add_random_small_phase:
        if verbose :
            print('add a small random phase in addition')
        small_phase_range = np.random.uniform(.1*pi, 2.*pi)
        obj_small_rand_phase = replace_phase_by_random_phase(obj, 
                                                      phase_range=small_phase_range,
                                                      correlation_length=corr_phase)
        phase_quad = phase_quad + np.angle(obj_small_rand_phase)
    
    # Apply phase range
    module = np.abs(obj)
    support = module > threshold_module * np.max(module)
    phase_quad = phase_quad * phase_range / np.ptp(phase_quad[support])
    
    # Create object with quadratic phase
    obj_new_phase = np.abs(obj) * np.exp(1.0j*(phase_quad))
    
    # remove any possible phase ramp
    obj_new_phase  = remove_phase_ramp(obj_new_phase ,
                            threshold_module=.3,
                            crop=False,
                            return_ramp=False,
                            method='fit', # 'gradient'
                            plot=False) 

    if plot:
        plot_2D_slices_middle(obj_new_phase, fig_title='object with replaced phase')
        
    if verbose :
        print('quadratic along axis :', axis_quadratic)
        print(f'phase_range : {phase_range/pi} * pi', )
    return obj_new_phase

######################################################################################################################################
################################            Poisson Noise            ###########################################################
######################################################################################################################################

def force_poisson_statistic(obj, 
                            scale_poisson=None,
                            seed = False, 
                            plot=False):

    if seed:
        np.random.seed(42)
    
    if scale_poisson is None:
        scale_poisson_power = np.random.uniform(3.8, 5.5)  # CHOSEN BY EB. Don't hesitate to change
        scale_poisson = 10**scale_poisson_power
        
    Fexp = ifftshift(fftn(fftshift(obj))) # diffracted complex amplitude
    I = np.abs(Fexp)**2. # diffracted intensity 
            
    I_poisson = np.random.poisson(lam = I * scale_poisson / np.max(I)).astype('float64') # apply poisson statistic
    phi = np.angle(Fexp)
       
    if plot:
        plot_diffraction_3d(I, fig_title='diffracted intensity')
        plot_2D_slices_middle(obj, threshold_module=.3, fig_title='corresponding object')
        plot_diffraction_3d(I_poisson, fig_title='diffracted intensity with poisson statistic')
        plot_2D_slices_middle(obj_poisson, threshold_module=.3, fig_title='corresponding object after poisson statistic')
        
    return I_poisson, phi



######################################################################################################################################
################################            Overall function that changes phase and adds noise         ########################################################
######################################################################################################################################
def add_random_noise(obj, 
                     strain = None,
                     corr_phase = None,
                     phase_range1 = None,
                     phase_range2 = None, 
                     sigma1 = None, 
                     sigma2 = None, 
                     poisson_noise = True,
                     scale_poisson = None,
                     seed = False,  
                     plot=False):
    
    if seed:
        np.random.seed(42)
        
    if plot:
        Fexp = ifftshift(fftn(fftshift(obj))) # diffracted complex amplitude
        I = np.abs(Fexp)**2
        plot_diffraction_3d(I, fig_title='diffracted intensity')
        plot_2D_slices_middle(obj, fig_title='real space object', threshold_module=.3)
    
    # Smooth real space module
    obj = smooth_object(obj, seed, sigma_gaussian=None)
    obj = remove_real_space_module_out_support(obj, threshold_module=.1)
    
    # add random noise to the real space module
    module = np.abs(obj)
    module_noise = add_random_noise_module(module, seed, module_factor = None, correlation_length=None)
    obj = module_noise * np.exp(1.0j*np.angle(obj))
    
    if corr_phase is None:
        # Calculate modulus of the complex object
        modulus = np.abs(obj)

        # Compute the support mask (non-zero modulus)
        support_mask = modulus > .3*np.max(modulus)

        # Calculate volume of the support (sum of non-zero values)
        support_volume = np.sum(support_mask)

        corr_phase = 1/(np.mean(compute_oversampling_ratio(obj))*2.4)
        
        
    if strain == 'random':
        
        obj = replace_phase_by_random_phase(obj, phase_range=phase_range1, correlation_length=corr_phase)
        obj = remove_phase_ramp(obj,
                      threshold_module=.3,
                      crop=False,
                      return_ramp=False,
                      method='fit', # 'gradient'
                      plot=False) 
        
        
    if strain == 'gauss':
        obj = simulate_strain_gauss(obj, phase_range1 = phase_range1, phase_range2 = phase_range2, sigma1 = sigma1, sigma2 = sigma2)
        obj = remove_phase_ramp(obj,
                      threshold_module=.3,
                      crop=False,
                      return_ramp=False,
                      method='fit', # 'gradient'
                      plot=False) 
        
    if strain == 'cosine':
        obj = simulate_strain_cosine(obj, phase_range1 = phase_range1, phase_range2 = phase_range2, sigma1 = sigma1, sigma2 = sigma2)
        obj = remove_phase_ramp(obj,
                      threshold_module=.3,
                      crop=False,
                      return_ramp=False,
                      method='fit', # 'gradient'
                      plot=False)
        
    if strain == 'quadratic':
        obj = simulate_strain_quad_substrate(obj, phase_range = phase_range1)

    
    if poisson_noise is False:
        Fexp = ifftshift(fftn(fftshift(obj)))
        I = np.abs(Fexp)**2
        phi = np.angle(Fexp)
    
    else:
        I, phi = force_poisson_statistic(obj, seed = seed, scale_poisson=scale_poisson)

    if plot:
        
        plot_diffraction_3d(I, fig_title = 'diffracted intensity after random noises')
        plot_2D_slices_middle(obj, fig_title ='real space object after random noises', threshold_module=.3)

    return I, phi

############################################################################################################################################
###########################################      Calculate Obj center of mass       #######################################
############################################################################################################################################

import numpy as np
from scipy.ndimage import fourier_shift

def calculate_center_of_mass(object_3d):
    """
    Calculates the center of mass of the 3D object based on its magnitude.
    
    Parameters:
    - object_3d (ndarray): A 3D complex object.
    
    Returns:
    - center_of_mass (ndarray): The (x, y, z) coordinates of the center of mass.
    """
    # Use the magnitude of the complex numbers to calculate the center of mass
    magnitudes = np.abs(object_3d)
    
    # Create coordinate grids
    z, y, x = np.indices(object_3d.shape)
    
    # Calculate the weighted sum of coordinates
    total_mass = np.sum(magnitudes)
    x_center_of_mass = np.sum(x * magnitudes) / total_mass
    y_center_of_mass = np.sum(y * magnitudes) / total_mass
    z_center_of_mass = np.sum(z * magnitudes) / total_mass
    
    return np.array([x_center_of_mass, y_center_of_mass, z_center_of_mass])

############################################################################################################################################
###########################################      Center a 3d object according to the center of mass     #######################################
############################################################################################################################################

def center_object(object_3d):
    """
    Shifts the 3D complex object such that its center of mass is at the center of the array.
    
    Parameters:
    - object_3d (ndarray): A 3D complex object to be centered.
    
    Returns:
    - centered_object_3d (ndarray): The 3D object with its center of mass shifted to the center.
    """
    # Calculate the center of mass of the 3D object
    center_of_mass = calculate_center_of_mass(object_3d)
    
    # Get the center of the array (assuming object_3d is cubic)
    array_center = np.array(object_3d.shape) / 2
    
    # Calculate the shift vector (difference between array center and center of mass)
    shift_vector = array_center - center_of_mass
    
    # Perform Fourier-based shift to avoid artifacts
    centered_object_3d = np.fft.ifftn(fourier_shift(np.fft.fftn(object_3d), shift_vector))
    
    return centered_object_3d


############################################################################################################################################
###########################################      PLOT 3D CENTRAL SLICES OF DIFFRACTION PATTERN       #######################################
############################################################################################################################################


def plot_diffraction_3d(I, fig_title=None):
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(np.log(I[I.shape[0]//2]))
    ax[0].set_title('slice in the middle\nof the 1$^{st}$ dimension', fontsize=15)
    ax[1].imshow(np.log(I[:,I.shape[1]//2]))
    ax[1].set_title('slice in the middle\nof the 2$^{nd}$ dimension', fontsize=15)
    ax[2].imshow(np.log(I[:,:,I.shape[2]//2]))
    ax[2].set_title('slice in the middle\nof the 3$^{rd}$ dimension', fontsize=15)
    
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=20)
        
    fig.tight_layout()
    return