import numpy as np
from numpy import sqrt

def RotationMatrix(orientation):
    '''
    1 : x2X
    2 : a2A
    3 : b2B   
    4 : c2C   
    5 : d2D   
    6 : e2E   
    7 : f2F   
    8 : g2G
    9 : h2H
    10 : i2I
    11 : j2J
    12 : k2K
    '''
    
    if orientation==0: # x2X
        rotation = np.array([ [1/sqrt(2), -1/sqrt(2) , 0],
                         [1/sqrt(6), 1/sqrt(6), -2/sqrt(6)],
                         [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)] ])
        
    if orientation==1: # a2A
        rotation = np.array([ [0, 1./sqrt(2.), -1./sqrt(2.)],
                         [-2./sqrt(6.), 1./sqrt(6.), 1./sqrt(6.)],
                         [1./sqrt(3.), 1./sqrt(3.), 1./sqrt(3.)] ])
    if orientation==2: # b2B
        rotation = np.array([ [-1./sqrt(2.) ,  0, 1./sqrt(2.)],
                         [1./sqrt(6.), -2./sqrt(6.), 1./sqrt(6.)],
                         [1./sqrt(3.), 1./sqrt(3.), 1./sqrt(3.)] ])
        
    if orientation==3: # c2C
        rotation = np.array([ [1./sqrt(2.), 0, 1./sqrt(2.)],
                         [1./sqrt(6.), 2./sqrt(6.), -1./sqrt(6.)],
                         [-1./sqrt(3.), 1./sqrt(3.), 1./sqrt(3.)] ])
        
    if orientation==4: # d2D
        rotation = np.array([ [1./sqrt(2.) , 1./sqrt(2.), 0],
                         [-1./sqrt(6.), 1./sqrt(6.), -2./sqrt(6.)],
                         [-1./sqrt(3.), 1./sqrt(3.), 1./sqrt(3.)] ]) 
        
    if orientation==5: # e2E
        rotation = np.array([ [0, 1./sqrt(2.), -1./sqrt(2.)],
                         [-2./sqrt(6.),  -1./sqrt(6.), -1./sqrt(6.)],
                         [-1./sqrt(3.),  1./sqrt(3.),  1./sqrt(3.)] ])  
        
    if orientation==6: # f2F
        rotation = np.array([ [1./sqrt(2.),  1./sqrt(2.),  0],
                         [-1./sqrt(6.), 1./sqrt(6.), 2./sqrt(6.)],
                         [1./sqrt(3.), -1./sqrt(3.), 1./sqrt(3.)] ])   
        
    if orientation==7: # g2G
        rotation = np.array([ [1./sqrt(2.), 0, -1./sqrt(2.)],
                         [1./sqrt(6.), 2./sqrt(6.), 2./sqrt(6.)],
                         [1./sqrt(3.), -1./sqrt(3.), 1./sqrt(3.)] ])  
        
    if orientation==8: # h2H
        rotation = np.array([ [0,  1./sqrt(2.),  1./sqrt(2.)],
                         [-2./sqrt(6.), -1./sqrt(6.), 1./sqrt(6.)],
                         [1./sqrt(3.), -1./sqrt(3.), 1./sqrt(3.)] ]) 
        
    if orientation==9: # i2I
        rotation = np.array([ [0, 1./sqrt(2.), 1./sqrt(2.)],
                         [2./sqrt(6.), -1./sqrt(6.), 1./sqrt(6.)],
                         [1./sqrt(3.), 1./sqrt(3.), -1./sqrt(3.)] ]) 
        
    if orientation==10: # j2J
        rotation = np.array([ [1./sqrt(2.), -1./sqrt(2.), 0.],
                         [-1./sqrt(6.), -1./sqrt(6.), -2./sqrt(6.)],
                         [ 1./sqrt(3.), 1./sqrt(3.), -1./sqrt(3.)] ]) 
        
    if orientation==11: # k2K
        rotation = np.array([ [1./sqrt(2.), 0, 1./sqrt(2.)],
                         [1./sqrt(6.), -2./sqrt(6.), -1./sqrt(6.)],
                         [ 1./sqrt(3.), 1./sqrt(3.), -1./sqrt(3.)] ])    
        
    return rotation