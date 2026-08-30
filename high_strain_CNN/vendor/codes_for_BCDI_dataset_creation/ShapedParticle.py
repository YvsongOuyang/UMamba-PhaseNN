import numpy as np
import random
from numpy.random import randint, uniform
from Particle import *
import ase
import ase.cluster as cls

class ShapedParticle(Particle):
    'Cut the initial cubic particle to a given shape'
    def __init__(self, shape, shape_parameters=None, NL = [40, 40, 40], element='Au', path_potential = '/data/id01/inhouse/masto/3D_phasing_simulated/Main_Files/pot/',
                print_mode='info_shaped_particle', target_degree=0.5, number_of_cuts = 3):
        
        '''
        :shape: available shapes are 'winterbottom', 'wulff', 'octahedron', 'cube', 'random', 'roundish'
        :shape_parameters: give the shape_parameters if you want to construct a non-random particle.
         Careful, the parameters depends on the shape (shape_parameters is a dictionary)
        :NL: array of 3 elements giving the size of the particle in lattice parameter units
        :element: atomic element
        :path_potential: path of the potential containing some informations about the element
        :print_info: 'info_shaped_particle' if you want to print some informations
        '''
        
        super().__init__(NL=NL,  element=element, path_potential=path_potential,
                print_mode=print_mode)
        
        self.shape = shape
        self.shape_parameters = shape_parameters
        self.target_degree = target_degree
        self.nb_cuts = number_of_cuts

        if self.shape=='winterbottom':
            if self.shape_parameters is not None:
                self.WinterbottomFromParameterDict()
            else:
                self.Winterbottom()
                
        if self.shape=='wulff':
            if self.shape_parameters is not None:
                self.WulffFromParameterDict()
            else:
                self.Wulff()
            
        if self.shape=='octahedron':
            if self.shape_parameters is not None:
                self.OctahedronFromParameterDict()
            else:
                self.Octahedron()
                
        if self.shape=='cube':
            if self.shape_parameters is not None:
                self.CubeFromParameterDict()
            else:
                self.Cube()
                
        if self.shape=='random':
            self.RandomShape()
        
        if self.shape=='centrosymmetric':
            self.RandomShape_centrosymmetric()
            
        if self.shape =='random with symmetricity':
            self.RandomShape_with_symmetricity(target_degree = self.target_degree)

        if self.shape == 'tetrahedron':
            if self.shape_parameters is not None:
                self.TetrahedronFromParameterDict()
            else:
                self.Tetrahedron()

        # New shape: roundish
        if self.shape == 'roundish':
            if self.shape_parameters is not None:
                self.Roundish(relative_size=self.shape_parameters.get('relative_size', None))
                              # number_of_cuts=self.shape_parameters.get('number_of_cuts', None))
            else:
                self.Roundish()
           
        'Print some informations'
        if print_mode=='info_shaped_particle':
            self.PrintInformations()
        
        
    def CutAlongGivenDirection(self,direction, distance_from_center):
        '''
        Cut the particle with a plane along a given direction and with a given distance form the origin
        :direction: 3d vector giving the direction perpendicular to the plane cut
        :distance_from_center: distance of the plane cut from the origin
        '''
        direction = np.array(direction)
        direction = direction/np.sqrt(np.sum(direction**2.))
        self.u = np.delete(self.u, np.dot(self.u,direction) > distance_from_center, axis=0)
        return
    
            
    def CutAlongRandomDirection(self, distance_from_center):
        '''
        Cut the particle with a plane along a given random direction and with a given distance form the origin
        '''
        direction = np.random.randint(-6,6, size=3)
        direction = direction/np.sqrt(np.sum(direction**2.))
        self.u = np.delete(self.u, np.dot(self.u,direction) > distance_from_center, axis=0)
        return
    
    #####################################################################################################################################
    ###################################################      Winterbottom     ##########################################################
    ##################################################################################################################################### 
    
    def Winterbottom(self, cut_p=None, ratio_111=None, ratio_110=None, bottom_surface_direction_index=None):
        '''
        :cut_p: define the cut on the surface that is in contact with the subtrate
        :ratio_111: energy ratio of the 111 surfaces
        :ratio_110: energy ratio of the 110 surfaces
        :bottom_surface_direction_index: an integer between 0 and 7
        '''
        
        # If the parameters are not given by the user, we choose them randomly
        if cut_p is None:
            ref = 0.31 ;
            rand = 0.02*randint(1, 10+1)
            cut_p = ref + rand 
        if ratio_111 is None:
            ratio_111 = 0.923+.05*uniform(-1,1)
        if ratio_110 is None:
            ratio_110 = 1.183+.05*uniform(-1,1) 
        if bottom_surface_direction_index is None :
            bottom_surface_direction_index = np.random.randint(8)
            
        
        if bottom_surface_direction_index not in np.arange(8):
            print('Error, bottom_surface_direction_index should be between 0 and 7')
            
        L = self.NL[0]*self.a

        # Cuts along 111 type of planes
        directions_111 = np.array([[1,1,1], [1,-1,1], [-1,1,1],  [1,1,-1],
                              [-1,-1,-1], [-1,1,-1], [1,-1,-1],  [-1,-1,1]])
        bottom_surface_direction = directions_111[bottom_surface_direction_index]
        direction_111 = np.delete(directions_111, bottom_surface_direction_index,axis=0)
        
        if self.print_mode=='info':
            print('winterbottom bottom surface direction :', bottom_surface_direction)
        
        
        for direction in directions_111:
            self.CutAlongGivenDirection(direction, ratio_111*L/2.) 
        self.CutAlongGivenDirection(bottom_surface_direction, cut_p*ratio_111*L/2.)
        
        # Cut along the 11 type of planes
        for direction in np.array([ [1,1,0], [-1,1,0], [1,0,1], [-1,0,1], [0,1,1], [0,-1,1] ]):
            self.CutAlongGivenDirection(direction, ratio_110*L/2.)
            self.CutAlongGivenDirection(-direction, ratio_110*L/2.)
            
        self.shape_parameters = {'cut_p' : cut_p,
                                 'ratio_111' : ratio_111,
                                 'ratio_110' : ratio_110,
                                 'bottom_surface_direction_index' : bottom_surface_direction_index,
                                 'bottom_surface_direction' : bottom_surface_direction}

        return 
    
    def WinterbottomFromParameterDict(self):
        '''
        Construct a non-random winterbottom particle if shape_parameter is given by the user
        '''
        cut_p = self.shape_parameters['cut_p']
        ratio_111 = self.shape_parameters['ratio_111']
        ratio_110 = self.shape_parameters['ratio_110']
        bottom_surface_direction_index = self.shape_parameters['bottom_surface_direction_index']
        self.Winterbottom(cut_p=cut_p, ratio_111=ratio_111, ratio_110=ratio_110,
                               bottom_surface_direction_index=bottom_surface_direction_index)
        return
    
    #####################################################################################################################################
    ######################################################      Wulff     #############################################################
    ##################################################################################################################################### 

    def Wulff(self, size=None, surfaces=None, energies=None):
        '''
        This function takes a long time to compute compared to the winterbottom creation.
        It actually doesn't start from the cubic particle but gives directly the wulff shape using the ASE module.
        '''
    
        # If the parameters are not given by the user, we choose them randomly
        if size is None:
            size = random.randint(80000, 140000)
            
        if surfaces is None:
            miller_indexes = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                              (1, 1, 0), (1, 0, 1), (0, 1, 1),
                              (-1, 1, 0), (-1, 0, 1), (0, -1, 1),
                              (1, 1, 1), (-1, 1, 1), (1, -1, 1),
                              (1, 1, -1), (1, 0, 2), (-1, 0, 2)]
            k = random.randint(1, 14)
            surfaces = random.sample(miller_indexes, k=k)
            
        if energies is None:
            energies = [random.uniform(0.9, 1.1) for _ in range(len(surfaces))] 
            
        atoms = cls.wulff_construction(self.element, surfaces, energies, size, structure='fcc')
        self.u = atoms.positions
        
        self.shape_parameters = {'size' : size,
                                 'surfaces' : surfaces,
                                 'energies' : energies}
        return
    
    
    def WulffFromParameterDict(self):
        '''
        Construct a non-random wulff particle if shape_parameter is given by the user
        '''
        size = self.shape_parameters['size']
        surfaces = self.shape_parameters['surfaces']
        energies = self.shape_parameters['energies']
        self.Wulff(size=size, surfaces=surfaces, energies=energies)
        return
    
    
    #####################################################################################################################################
    ####################################################      Octahedron     ############################################################
    ##################################################################################################################################### 
    
    
    def Octahedron(self, length=None):
        '''
        It actually doesn't start from the cubic particle but gives directly the octahedron shape using the ASE module.
        '''
        if length is None:
            length = random.randint(45, 60)

        atoms = cls.Octahedron(self.element, length)
        self.u = atoms.positions

        self.shape_parameters = {'length' : length}
        return 
    
    def OctahedronFromParameterDict(self):
        '''
        Construct a non-random octahedron particle if shape_parameter is given by the user.
        '''
        length = self.shape_parameters['length']
        self.Octahedron(length)
        return
    #####################################################################################################################################
    ####################################################      Tetrahedron     ############################################################
    ##################################################################################################################################### 
    
    
    def Tetrahedron(self, length=None):
        '''
        Creates a Tetrahedron by cutting an Octahedron into two using the ASE module.
        '''
        if length is None:
            length = random.randint(45, 60)

        # Generate the octahedron
        atoms = cls.Octahedron(self.element, length)

        # Find the centroid of the octahedron
        centroid = atoms.positions.mean(axis=0)

        # Keep only the atoms on one side of the centroid (cutting the octahedron in half)
        tetrahedron_atoms = atoms[[i for i, pos in enumerate(atoms.positions) if pos[2] > centroid[2]]]

        self.u = tetrahedron_atoms.positions

        self.shape_parameters = {'length': length}
        return 

    
    
    #####################################################################################################################################
    ######################################################      Cube     ################################################################
    ##################################################################################################################################### 

    def Cube(self, relative_size=None):
        '''
        Randomly cut the initial cubic particle into a smaller cube.
        :relative_size: float between 0 and 1.
        '''
        if relative_size is None:
            relative_size = random.uniform(.40,.92)
            
        cut_directions = [[1,0,0], [0,1,0], [0,0,1],
                         [-1,0,0], [0,-1,0], [0,0,-1]]
        for direction in cut_directions:
            self.CutAlongGivenDirection(direction, relative_size*self.L[0]/2.) 
            
        self.shape_parameters = {'relative_size' : relative_size}
        return 
    
    def CubeFromParameterDict(self):
        '''
        Construct a non-random cubic particle if shape_parameter is given by the user.
        '''
        relative_size = self.shape_parameters['relative_size']
        self.Cube(relative_size=relative_size)
        return
    
    #####################################################################################################################################
    ######################################################      Random    ################################################################
    #####################################################################################################################################
    
    def RandomShape(self, number_of_cuts = None):
        '''
        Randomly cut the initial cubic particle.
        :number_of_cuts: integer number of cuts.
        '''
        if number_of_cuts is None:
            number_of_cuts = np.random.randint(3,9)
        
        for direction in range(number_of_cuts):
            relative_size = random.uniform(.40,.92)
            self.CutAlongRandomDirection(relative_size*self.L[0]/2.)    
        
        return 
    
    def RandomShape_centrosymmetric(self, number_of_cuts=None):
        '''
        Randomly cut the initial cubic particle with centrosymmetry.
        '''
        if number_of_cuts is None:
            number_of_cuts = np.random.randint(3, 9)

        for _ in range(number_of_cuts):
            # Generate a random direction for the cut
            direction = np.random.randint(-6, 6, size=3)
            direction = direction / np.sqrt(np.sum(direction**2.))

            # Generate a random distance from the center for the cut
            relative_size = random.uniform(.40, .92)
            distance_from_center = relative_size * self.L[0] / 2.

            # Cut in the original direction
            self.CutAlongGivenDirection(direction, distance_from_center)

            # Cut in the mirrored direction to ensure centrosymmetry
            mirrored_direction = -direction
            self.CutAlongGivenDirection(mirrored_direction, distance_from_center)

        return
    
    def calculate_volume(self):
        return len(self.u)
    
    def CutAlongDirection(self, direction, distance_from_center):
        direction = np.array(direction) / np.sqrt(np.sum(direction ** 2.))
        self.u = np.delete(self.u, np.dot(self.u, direction) > distance_from_center, axis=0)

    def calculate_circumscribing_sphere_volume(self):
        max_radius = np.max(np.linalg.norm(self.u, axis=1))
        return (4/3) * np.pi * (max_radius ** 3)

    def centrosymmetricity_degree(self):
        particle_volume = self.calculate_volume()
        sphere_volume = self.calculate_circumscribing_sphere_volume()
        return particle_volume / sphere_volume if sphere_volume != 0 else 0

    def RandomShape_with_symmetricity(self, target_degree, tolerance=0.05, max_attempts=100):
        attempt = 0
        directions = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                      np.array([1, 1, 1]), np.array([-1, -1, 1]), np.array([1, -1, -1])]
        while attempt < max_attempts:
            # Reset particle points for each attempt
            self.u = self.generate_initial_points()
            
            # Apply initial cuts
            for direction in directions:
                distance = random.uniform(0.5, self.L[0] / 2)
                self.CutAlongDirection(direction, distance)
            
            # Calculate centrosymmetricity degree
            degree = self.centrosymmetricity_degree()
            print(degree)

            # Fine-tune based on comparison with target
            if abs(degree - target_degree) <= tolerance:
                print(f"Target centrosymmetricity degree achieved: {degree}")
                return degree
            
            if degree > target_degree:
                for direction in directions:
                    self.CutAlongDirection(-direction, distance)
            else:
                for direction in directions:
                    distance = random.uniform(0.1, self.L[0] / 4)
                    self.CutAlongDirection(direction, distance)

            attempt += 1

        print("Failed to achieve desired centrosymmetricity degree within the maximum attempts.")
        return None

    # def generate_initial_points(self):
    #     # Example function to regenerate initial points in a cubic structure
    #     grid_size = int(self.L[0] / 2)
    #     return np.array([[x, y, z] for x in range(-grid_size, grid_size)
    #                                for y in range(-grid_size, grid_size)
    #                                for z in range(-grid_size, grid_size)])
    def generate_initial_points(self):
        # Generate a cubic lattice of points, centered at origin
        xs = np.linspace(-self.L[0]/2, self.L[0]/2, self.NL[0])
        ys = np.linspace(-self.L[1]/2, self.L[1]/2, self.NL[1])
        zs = np.linspace(-self.L[2]/2, self.L[2]/2, self.NL[2])
        return np.array([[x, y, z] for x in xs for y in ys for z in zs])

    #####################################################################################################################################
    ######################################################    Roundish    #############################################################
    #####################################################################################################################################
    
    # def Roundish(self, radius=None, number_of_cuts=None):
    #     '''
    #     Creates a roundish particle by generating a spherical collection of points and then applying random cuts to form facets.
    #     :param radius: The radius of the initial sphere. If None, defaults to half the length of the particle in the x-direction.
    #     :param number_of_cuts: Number of random planar cuts to perform. If None, a random integer between 3 and 9 is chosen.
    #     '''
    #     # Determine a default radius if not provided
    #     if radius is None:
    #         radius = (self.NL[0]) / 2.0
        
    #     if number_of_cuts is None:
    #         number_of_cuts = random.randint(5, 9)
        
    #     # Generate a cubic grid and select points inside the sphere of the given radius
    #     grid = self.generate_initial_points()
    #     self.u = np.array([pt for pt in grid if np.linalg.norm(pt) <= radius])
        
    #     # # Apply random planar cuts to create facets
    #     # for _ in range(number_of_cuts):
    #     #     cut_distance = random.uniform(0.85 * radius, 0.95 * radius)
    #     #     self.CutAlongRandomDirection(cut_distance)
        
    #     self.shape_parameters = {'radius': radius, 'number_of_cuts': number_of_cuts}
    #     return
        
    def Roundish(self, relative_size=None):
        '''
        Construct a spherical particle filled with points inside the given radius.
        :relative_size: float between 0 and 1.
        '''
        if relative_size is None:
            relative_size = random.uniform(.40, .92)
    
        # Sphere radius relative to half the box size in x
        radius = relative_size * (self.L[0] / 2.0)
    
        # Generate a cubic grid centered at origin (or however generate_initial_points() is defined)
        grid = self.generate_initial_points()
    
        # Keep all points inside the sphere (solid ball, not just shell)
        self.u = np.array([pt for pt in grid if np.linalg.norm(pt) <= radius])
    
        self.shape_parameters = {'relative_size': relative_size}
        return
    
    
    def SphereFromParameterDict(self):
        '''
        Construct a non-random spherical particle if shape_parameter is given by the user.
        '''
        relative_size = self.shape_parameters['relative_size']
        self.Sphere(relative_size=relative_size)
        return


