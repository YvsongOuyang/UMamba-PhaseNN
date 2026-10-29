"""One rigid-registration convention for training and evaluation.

Each field is normalized by its own whole-image RMS. Amplitude-COM centering
provides an initial translation. A bounded rigid translation and one global
phase are fitted for BOTH direct and conjugate-inverted predictions. The same
minimum whole-image complex-L2 branch is used by every reported metric.

Registration decisions are detached. Gradients flow through the selected rigid
transform and complex field, not through the discrete search. Target information
is used only in the supervised objective/evaluation, never model inference.
"""
from __future__ import annotations
import math
import torch

OBJECTIVE_VERSION = 'rigid_complex_nrmse_v7'


def shift_field(field, shift_y, shift_x):
    """Periodic Fourier translation of complex [B,1,H,W] fields."""
    h,w=field.shape[-2:]
    fy=torch.fft.fftfreq(h,device=field.device,dtype=field.real.dtype)
    fx=torch.fft.fftfreq(w,device=field.device,dtype=field.real.dtype)
    angle=-2*math.pi*(shift_y[:,None,None,None]*fy[None,None,:,None]
                       +shift_x[:,None,None,None]*fx[None,None,None,:])
    return torch.fft.ifft2(torch.fft.fft2(field)*torch.exp(torch.complex(torch.zeros_like(angle),angle)))


def _center(field):
    # COM is a registration decision, held fixed during differentiation.
    with torch.no_grad():
        amplitude=field.detach().abs().to(torch.float64)
        mass=amplitude.sum((-2,-1))[:,0].clamp_min(1e-15)
        h,w=field.shape[-2:]
        ys=torch.arange(h,device=field.device,dtype=torch.float64)
        xs=torch.arange(w,device=field.device,dtype=torch.float64)
        cy=(amplitude*ys[None,None,:,None]).sum((-2,-1))[:,0]/mass
        cx=(amplitude*xs[None,None,None,:]).sum((-2,-1))[:,0]/mass
        dy=(h//2-cy).to(field.real.dtype)
        dx=(w//2-cx).to(field.real.dtype)
    return shift_field(field,dy,dx)


@torch.no_grad()
def _search(prediction,target,radius):
    """Batched coarse-to-fine DFT correlation; no CPU image copies."""
    cross=torch.fft.fft2(prediction[:,0])*torch.fft.fft2(target[:,0]).conj()
    b,h,w=cross.shape
    fy=torch.fft.fftfreq(h,device=cross.device,dtype=cross.real.dtype)
    fx=torch.fft.fftfreq(w,device=cross.device,dtype=cross.real.dtype)
    best=cross.sum((-2,-1))/(h*w)
    best_y=torch.zeros(b,device=cross.device,dtype=cross.real.dtype)
    best_x=torch.zeros_like(best_y)
    for step,reach in ((.5,radius),(.1,.5),(.02,.1),(.005,.02)):
        n=round(reach/step)
        offset=torch.arange(-n,n+1,device=cross.device,dtype=cross.real.dtype)*step
        ys=(best_y[:,None]+offset[None,:]).clamp(-radius,radius)
        xs=(best_x[:,None]+offset[None,:]).clamp(-radius,radius)
        ay=-2*math.pi*ys[:,:,None]*fy[None,None,:]
        ax=-2*math.pi*xs[:,:,None]*fx[None,None,:]
        ey=torch.exp(torch.complex(torch.zeros_like(ay),ay))
        ex=torch.exp(torch.complex(torch.zeros_like(ax),ax))
        correlations=torch.bmm(torch.bmm(ey,cross),ex.transpose(1,2))/(h*w)
        index=correlations.abs().flatten(1).argmax(1)
        chosen=correlations.flatten(1).gather(1,index[:,None])[:,0]
        iy=torch.div(index,xs.shape[1],rounding_mode='floor')
        ix=index % xs.shape[1]
        better=chosen.abs()>best.abs()
        best_y=torch.where(better,ys.gather(1,iy[:,None])[:,0],best_y)
        best_x=torch.where(better,xs.gather(1,ix[:,None])[:,0],best_x)
        best=torch.where(better,chosen,best)
    return best_y,best_x,best


def align_objects(prediction,target,search_radius=4.):
    if (prediction.ndim!=4 or prediction.shape[1]!=1 or prediction.shape!=target.shape
            or not torch.is_complex(prediction) or not torch.is_complex(target)):
        raise ValueError('Expected equal-shaped complex [B,1,H,W] prediction and target.')
    if prediction.dtype!=target.dtype or prediction.device!=target.device:
        raise ValueError('Prediction/target dtype and device must agree.')
    if not math.isfinite(search_radius) or not 0<search_radius<=8:
        raise ValueError('Require finite 0 < search_radius <= 8.')
    if not bool(torch.isfinite(prediction).all() & torch.isfinite(target).all()):
        raise ValueError('Nonfinite complex object.')
    rms_p=prediction.abs().square().mean((-2,-1),keepdim=True).sqrt()
    rms_t=target.abs().square().mean((-2,-1),keepdim=True).sqrt()
    if bool((rms_p<=1e-12).any() | (rms_t<=1e-12).any()):
        raise ValueError('Cannot normalize a zero-energy complex object.')
    p,t=_center(prediction/rms_p),_center(target/rms_t)
    twin=torch.conj(torch.roll(torch.flip(p,(-2,-1)),(1,1),(-2,-1)))
    b=len(p)
    with torch.no_grad():
        dy,dx,corr=_search(torch.cat((p.detach(),twin.detach())),torch.cat((t.detach(),t.detach())),search_radius)
        select_twin=corr[b:].abs()>corr[:b].abs()
        y=torch.where(select_twin,dy[b:],dy[:b])
        x=torch.where(select_twin,dx[b:],dx[:b])
        c=torch.where(select_twin,corr[b:],corr[:b])
        phase=-torch.angle(c)
    selected=torch.where(select_twin[:,None,None,None],twin,p)
    aligned=shift_field(selected,y,x)*torch.exp(torch.complex(torch.zeros_like(phase),phase))[:,None,None,None]
    info={'shift_y_px':y,'shift_x_px':x,'selected_twin':select_twin.to(p.real.dtype),
          'search_boundary_hit':((y.abs()>=search_radius-.0051)|(x.abs()>=search_radius-.0051)).to(p.real.dtype)}
    return aligned,t,info


def objective_and_metrics(prediction,target,search_radius=4.):
    p,t,info=align_objects(prediction,target,search_radius)
    energy=t.abs().square().flatten(1).sum(1).clamp_min(1e-12)
    relative_error=(p-t).abs().square().flatten(1).sum(1)/energy
    # Stable differentiable floor; reporting retains the actual unfloored score.
    loss=(relative_error+1e-12).sqrt().mean()
    with torch.no_grad():
        amp_p,amp_t=p.abs(),t.abs()
        amp_difference=amp_p-amp_t
        mask=amp_t/amp_t.amax((-2,-1),keepdim=True).clamp_min(1e-12)>=.3
        phase_error=torch.angle(p*t.conj()).abs()
        # All diagnostics reuse the exact same aligned objects; no separate fits.
        fp=torch.fft.fftshift(torch.fft.fft2(p),dim=(-2,-1))
        ft=torch.fft.fftshift(torch.fft.fft2(t),dim=(-2,-1))
        keep=torch.ones_like(ft.real)
        cy,cx=t.shape[-2]//2,t.shape[-1]//2
        keep[...,cy-4:cy+5,cx-4:cx+5]=0
        values={
            'registered_complex_nrmse':relative_error.sqrt(),
            'amplitude_relative_l2_rms':(amp_difference.square().flatten(1).sum(1)/energy).sqrt(),
            'amplitude_relative_l1_rms':amp_difference.abs().flatten(1).sum(1)/amp_t.flatten(1).sum(1).clamp_min(1e-12),
            'phase_mae_true_support':(phase_error*mask).flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1),
            'offaxis_nrmse_9x9':(((fp-ft).abs().square()*keep).flatten(1).sum(1)/(ft.abs().square()*keep).flatten(1).sum(1).clamp_min(1e-12)).sqrt(),
            **info,
        }
    return loss,values
