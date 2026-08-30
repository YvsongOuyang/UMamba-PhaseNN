import numpy as np

class Particle():
    'Create a simulated particule'
    def __init__(self, NL = [40, 40, 40], element='Au', path_potential = '/data/id01/inhouse/masto/3D_phasing_simulated/Main_Files/pot/',
                print_mode='info_particle'):
        '''
        :NL: array of 3 elements giving the size of the particle in lattice parameter units
        :element: atomic element
        :path_potential: path of the potential containing some informations about the element
        :print_info: 'info_particle' if you want to print some informations
        '''

        'Initialization'
        self.path_potential = path_potential
        self.element = element
        self.NL = NL
        self.print_mode = print_mode
        self.InitializationParamteters()

                
        'Create the FCC lattice'
        # the atomic positions are stored in in a new attribute called "u" 
        # (yes the name isn't great but it's short and we call it a lot)
        self.CreateCubicCrystalFCC()
        
        
        'Print some informations'
        if print_mode=='info_particle':
            self.PrintInformations()
        
        
        
#####################################################################################################################################
#############################################      Initialization functions     #######################################################
#####################################################################################################################################


    def PrintInformations(self, dictio=None, indent=0):
        if dictio is None:
            dictio=self.__dict__
        for key, value in dictio.items():
            print('\t' * indent + str(key), end='')
            if isinstance(value, dict):
                print('')
                self.PrintInformations(dictio=value, indent=indent+1)
            else:
                print(' : ' + str(value))
        return


    def GetPotentialParams(self,potential_filename):
        '''
        :potential_filename: path of the potential file
        :return: several parameters used for the simulated crystal
        '''
        potential = open(potential_filename)
        
        if self.print_mode=='info':
            print('potential file : ',potential_filename )

        for n in range(4):
            potential.readline()
        NAME0, MASS = potential.readline().split()
        MASS = float(MASS)
        a, useless = potential.readline().split()
        a = float(a)

        Rc = float(potential.readline())

        return NAME0, MASS, a, Rc
        
        
    def InitializationParamteters(self):
        # potential 
        potentials = {'Au' : 'GOLD/Au_GROCHOLA.eam',
                     'Pt' : 'Pt/Pt_Zhou04.eam'} # dictionnary of elements potentials
        potential_filename = self.path_potential+potentials[self.element]
        NAME0, MASS, a, Rc = self.GetPotentialParams(potential_filename)
    
        L = [None, None, None, None]
        L[0] = self.NL[0]*a#*self.cellsize[0]
        L[1] = self.NL[1]*a#*self.cellsize[1]
        L[2] = self.NL[2]*a#*self.cellsize[2]
        L[3] = 0. # No idea what it's used for
        
        # Add those as attributes to the object\
        setattr(self, 'L', L)
        setattr(self, 'a', a)
        setattr(self, 'MASS', MASS)
        setattr(self, 'Rc', Rc)
        
        return 
    
    
    
#####################################################################################################################################
#############################################      Create FCC cubic particle     ####################################################
##################################################################################################################################### 

    def unit_cell_FCC(self):
        '''
        Create an FCC unit cell.
        :return: U containing the 3D positions for the 4 atoms in the unit cell.
        '''
        U = np.zeros((4,3))

        U[1,0] = 1./2.
        U[1,1] = 1./2.
        U[1,2] = 0.

        U[2,0] = 1./2.
        U[2,1] = 0.
        U[2,2] = 1./2.

        U[3,0] = 0.
        U[3,1] = 1./2.
        U[3,2] = 1./2.

        return U

    
    def CreateCubicCrystalFCC(self):
        '''
        Create a FCC cubic crystal of size NL[0]xNL[1]xNL[2]xa^3
        
        :list of parameters:
        :NL: number of cells along the 3 directions
        :a: lattice parameter
        :cellsize: we were only using [1, 1, 1]
        :L: the crystal size along the 3 directions. This was directly calculated using NL and a in InitializationParamteters()
        :return: u an array of size (number of atoms,3) containing the 3D position for every atom 
        '''

        i = range(self.NL[0])
        j = range(self.NL[1])
        k = range(self.NL[2])
        i,j,k = np.meshgrid(i,j,k)

        u = np.zeros((4,)+i.shape+(3,))
        u[...,0] += i#*self.cellsize[0]
        u[...,1] += j#*self.cellsize[1]
        u[...,2] += k#*self.cellsize[2]

        U = self.unit_cell_FCC()
        for n in range(len(U)):
            u[n] += U[n]

        u *= self.a

        u[...,0] -= self.L[0]/2.
        u[...,1] -= self.L[1]/2.
        u[...,2] -= self.L[2]/2.

        u = u.reshape(u.shape[0]*u.shape[1]*u.shape[2]*u.shape[3],u.shape[4])

        setattr(self, 'u', u)
        return 
    
#####################################################################################################################################
#########################################      Save the particle as an lmp file     #################################################
##################################################################################################################################### 

    def SaveLmpFile(self, lmp_name, simulation_cell_size_factor = 1.1):
        '''
        :lmp_name: named of the saved file. For example 'particle.lmp'
        :u: array of atomic position
        :MASS: the atom mass (taken from the potential file)
        :element: string of the element. For example 'Au'
        :simulation_cell_size_factor: size of the simulation cell relative to the particle size
                                      this cell should be a bit larger than the particle for the LAMMPS relaxation
                                      use a value larger than 1
        '''

        file = open(lmp_name,'w')
        
        file.write(" # Number of Au = {:d}\n\n".format(len(self.u)))
        file.write('      {:d}  atoms\n'.format(len(self.u)))
        file.write('           1  atom types\n\n'.format(len(self.u)))

        # Define simulation cell size.
        # We might need to have it a bit larger than the particle since atoms will move during the relaxation
        xmin = simulation_cell_size_factor*np.min(self.u[:,0])
        xmax = simulation_cell_size_factor*np.max(self.u[:,0])
        ymin = simulation_cell_size_factor*np.min(self.u[:,1])
        ymax = simulation_cell_size_factor*np.max(self.u[:,1])
        zmin = simulation_cell_size_factor*np.min(self.u[:,2])
        zmax = simulation_cell_size_factor*np.max(self.u[:,2])
        file.write('{:16.8f} {:16.8f}  xlo xhi\n'.format(xmin, xmax))
        file.write('{:16.8f} {:16.8f}  ylo yhi\n'.format(ymin, ymax))
        file.write('{:16.8f} {:16.8f}  zlo zhi\n\n'.format(zmin, zmax))

        file.write('Masses\n\n')
        file.write('           1 {:16.8f}    # {}\n\n'.format(self.MASS, self.element))
        # This doesn't give the same space size as atomsk. I guess it's fine

        file.write('Atoms # atomic\n\n')

        for n, position in enumerate(self.u):
            file.write('         {:d}    1 {:16.8f} {:16.8f} {:16.8f}\n'.format(n+1, position[0], position[1], position[2]))

        file.close()

        return 