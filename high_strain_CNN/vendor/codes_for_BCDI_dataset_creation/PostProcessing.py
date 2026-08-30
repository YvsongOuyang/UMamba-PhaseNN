import numpy as np
import pylab as plt
from numpy.fft import fftshift, fft, ifft
from numpy import pi

from pynx.cdi import * # Might take a bit of time to import

from Plot_utilities import *
from Global_utilities import *
from Object_utilities import *

my_cmap = MIR_Colormap()

#########################################################################################################################################
#################################                    Remove linear phase ramp                  ##########################################
#########################################################################################################################################

from sklearn.linear_model import LinearRegression
def linear_fit(array):

    pos = np.indices(array.shape)
    f = array[np.logical_not(np.isnan(array))]
    pos = [ p[np.logical_not(np.isnan(array))] for p in pos]

    X = np.zeros((len(f),len(pos)))
    for n in range(len(pos)):
        X[:,n] += pos[n]
    reg = LinearRegression().fit(X, f)
    
    return reg 

def remove_phase_linear_fit(phase):
    reg = linear_fit(phase)
    pos = np.indices(phase.shape)
#     ramp = np.sum([reg.coef_[n]*pos[-n] for n in range(len(pos))], axis=0) + reg.intercept_
#     ramp = -ramp
    ramp = np.sum([reg.coef_[n]*pos[+n] for n in range(len(pos))], axis=0) + reg.intercept_
    return ramp

def remove_phase_ramp_gradient_average(phase):
    # Get the slope
    grad = np.array(np.gradient(phase))#EB_custom_gradient(phase)
    slope = np.nanmean(grad, axis=(1,2,3))
    
    pos = np.indices(phase.shape).astype('float64')
    ramp = np.sum(pos * slope[:,None,None,None], axis=0)    
    return ramp

def remove_phase_ramp(obj,
                      threshold_module=.3,
                      crop=False,
                      return_ramp=False,
                      method='fit', # 'gradient'
                      plot=False):
    
    module, phase = get_cropped_module_phase(obj, crop=crop, unwrap=True, threshold_module=threshold_module)
    
    if method=='fit':
        ramp = remove_phase_linear_fit(phase)
    elif method=='gradient':
        ramp = remove_phase_ramp_gradient_average(phase)
    else:
        raise ValueError('no ramp computation method given')
        
    _, phase_full = get_cropped_module_phase(obj, crop=crop, unwrap=True, threshold_module=0.)
    phase_no_ramp = phase_full - ramp
    phase_no_ramp -= np.nanmean(phase_no_ramp) # Just remove a phase constant

    obj_no_ramp = np.abs(obj)*np.exp(1.0j*phase_no_ramp)
    if plot:
        
        if obj.ndim==2:
            fig, ax = plt.subplots(2,2, figsize=(8,8))            
            plot_object_module_phase_2d(obj, fig=fig, ax=ax[:,0], crop=crop,
                                       threshold_module=threshold_module)
            plot_object_module_phase_2d(obj_no_ramp, fig=fig, ax=ax[:,1], crop=crop,
                                       threshold_module=threshold_module)

            ax[0,0].set_title('object', fontsize=20)
            ax[0,1].set_title('object without phase ramp', fontsize=20)
            ax[0,0].set_ylabel('module', fontsize=20)
            ax[1,0].set_ylabel('phase', fontsize=20)
            fig.tight_layout()
            
            module, phase = get_cropped_module_phase(obj, crop=crop, 
                                                     threshold_module=threshold_module)
            module_no_ramp, phase_no_ramp = get_cropped_module_phase(obj_no_ramp, crop=crop,
                                                                    threshold_module=threshold_module)
            plt.figure()
            plt.matshow(phase-phase_no_ramp, cmap='hsv')
            plt.colorbar()
            plt.title('phase ramp', fontsize=20)
            
        if obj.ndim==3:
            fig, ax = plt.subplots(4,3, figsize=(3*4,3*3))
            plot_2D_slices_middle_only_module(obj, fig=fig, ax=ax[0], crop=crop)
            plot_2D_slices_middle_only_phase(obj, fig=fig, ax=ax[1], threshold_module=threshold_module, crop=crop)
            plot_2D_slices_middle_only_phase(obj_no_ramp, fig=fig, ax=ax[2], threshold_module=threshold_module, crop=crop)
            
            fake_obj_ramp = np.abs(obj)*np.exp(1.0j*(ramp))
            plot_2D_slices_middle_only_phase(fake_obj_ramp, fig=fig, ax=ax[3], threshold_module=threshold_module, crop=crop)
            
            ax[0,0].set_ylabel('module', fontsize=20)
            ax[1,0].set_ylabel('phase', fontsize=20)
            ax[2,0].set_ylabel('phase - ramp', fontsize=20)
            ax[3,0].set_ylabel('ramp', fontsize=20)
            fig.tight_layout()           
    if return_ramp:
        return obj_no_ramp, ramp
    else:
        return obj_no_ramp
    
#########################################################################################################################################
##############              Remove very large linear phase ramp (on purposed off-centered Bragg peak)             #######################
#########################################################################################################################################

def FT_remove_large_ramp(obj, 
                         offsets = None,
                         plot=False):
    F_recon = ifftshift(fftn(fftshift(obj)))
    
    I_recon = np.abs(F_recon)**2.

    if plot:
        plot_3D_projections(I_recon)

    if offsets is None:
        I_recon, offsets = center_the_center_of_mass(I_recon, return_offsets=True)
    else:
        I_recon = np.roll(I_recon, offsets, axis=range(I_recon.ndim))
    
    if plot:
        plot_3D_projections(I_recon)

    F_recon = np.roll(F_recon, offsets, axis=range(F_recon.ndim)) 

    obj = ifftshift(ifftn(fftshift(F_recon)))
    return obj, offsets

