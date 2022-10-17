import emopt
from emopt.misc import NOT_PARALLEL
from emopt.adjoint_method import AdjointMethodPNF2D

# We define the desired Gaussian modes in a separate file
# from mode_data import Ez_Gauss, Hx_Gauss, Hy_Gauss

import numpy as np
from math import pi

# secondly we make the optimization part 

class SiliconGratingAM(AdjointMethodPNF2D):

    def __init__(self, sim, grating_etch, wg, substrate, y_ts, w_in, h_wg, H, Ng, Nc, eps_clad, mm_line):
        ## Initialize variables
        # [PART 5]
        # Call super class constructor--important!
        super(SiliconGratingAM, self).__init__(sim, step=1e-10)

        # save the variables for later
        self.grating_etch = grating_etch
        self.wg = wg
        self.substrate = substrate
        self.y_ts = y_ts
        self.w_in = w_in
        self.h_wg = h_wg
        self.H = H
        self.Ng = Ng
        self.Nc = Nc

        self.mm_line = mm_line
        self.current_fom = 0.0

        # desired Gaussian beam properties used in mode match
        theta = 8.0/180.0*pi
        match_w0 = 10.4/2.0
        match_center = 11.0

        ## Setup FOM fields
        # [PART 8]
        # Define the desired field profiles
        # We use a tilted Gaussian beam which approximates a fiber mode
        Ezm, Hxm, Hym = emopt.misc.gaussian_fields(mm_line.x-match_center, 0.0, 0.0, match_w0, theta, sim.wavelength, np.sqrt(eps_clad))
        self.mode_match = emopt.fomutils.ModeMatch([0,1,0], sim.dx, Ezm=Ezm, Hxm=Hxm, Hym=Hym)

    def update_system(self, params):
        ## given design parameter values, update the system
        # [PART 6]
        coeffs = params
        Nc = self.Nc

        h_etch = params[-3]
        h_BOX = params[-1]
        x0 = self.w_in + params[-2]

        # we will use the coeff of the fourier transforms of the grating as design parameters 
        # for the grating since we want a smoothly varying grating to create a gaussian beam
        """A fourier series representation for the grating dimensions is chosen
        because it forces the grating to evolve in a smooth and gradual way
        (which is to be expected based on our physical intuition.) We could
        alternatively parameterize the individual tooth widths and gap sizes.
        This generally works pretty well, as well."""

        fseries = lambda i, coeffs : \
          coeffs[0] + np.sum([coeffs[j] *np.sin(pi/2*i*j*1.0/self.Ng) \
                              for j in range(1,Nc)]) \
                    + np.sum([coeffs[Nc + j] * np.cos(pi/2*i*j*1.0/self.Ng) \
                              for j in range(0,Nc)])

        for i in range(self.Ng):
            w_etch = fseries(i, coeffs[0:2*Nc])
            period = fseries(i, coeffs[2*Nc:4*Nc])

            # update the rectangles
            self.grating_etch[i].width  = w_etch
            self.grating_etch[i].height = h_etch
            self.grating_etch[i].x0 = x0 + w_etch/2.0
            self.grating_etch[i].y0 = self.y_ts + self.h_wg/2.0 - h_etch/2.0

            x0 += period

           # update the BOX/Substrate
        h_subs = self.H/2.0 - self.h_wg/2.0 - h_BOX
        self.substrate.height = h_subs
        self.substrate.y0 = h_subs/2.0

        # update the width of the unetched grating
        w_in = x0
        self.wg.width = w_in
        self.wg.x0 = w_in/2.0

    def calc_f(self, sim, params):
        ## calculate the value of the figure of merit
        # [PART 9]
        # Get the fields which were recorded
        Ez, Hx, Hy = sim.saved_fields[0]

        # compute the mode match efficiency
        self.mode_match.compute(Ez=Ez, Hx=Hx, Hy=Hy)

        # we want to maximize the efficiency, so we minimize the negative of the efficiency
        self.current_fom = -self.mode_match.get_mode_match_forward(1.0)
        return self.current_fom

    def calc_dfdx(self, sim, params):
        ## calculate the derivatives of the figure of merit with respect to
        ## the electric and magnetic fields (df/dEz, df/dHx, and df/dHy)
        # [PART 10]
        dFdEz = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dFdHx = np.zeros([sim.M, sim.N], dtype=np.complex128)
        dFdHy = np.zeros([sim.M, sim.N], dtype=np.complex128)

        # Get the fields which were recorded
        Ez, Hx, Hy = sim.saved_fields[0]

        self.mode_match.compute(Ez=Ez, Hx=Hx, Hy=Hy)

        dFdEz[self.mm_line.j, self.mm_line.k] = -self.mode_match.get_dFdEz()
        dFdHx[self.mm_line.j, self.mm_line.k] = -self.mode_match.get_dFdHx()
        dFdHy[self.mm_line.j, self.mm_line.k] = -self.mode_match.get_dFdHy()
        return (dFdEz, dFdHx, dFdHy)

    def calc_grad_p(self, sim, params):
        ## calculate the derivative of the figure of merit with respect to
        ## the design parameters
        # We won't use this in this tutorial
        """Out figure of merit contains no additional non-field dependence on
        the design variables so we just return zeros here.

        See the AdjointMethod documentation for the mathematical details of
        grad y and to learn more about its use case.
        """
        return np.zeros(params.shape)

    def get_update_boxes(self, sim, params):
        ## define what regions of the simulation domain are affected by
        ## changes to the design variables
        # [PART 7]
        h_wg = self.h_wg
        y_wg = self.y_ts
        lenp = len(params)

        # define boxes surrounding grating
        boxes = [(0, sim.X, y_wg-h_wg, y_wg+h_wg) for i in range(lenp-1)]

        # for BOX, update everything (easier)
        boxes.append((0, sim.X, 0, sim.Y))
        return boxes


def plot_update(params, fom_list, sim, am):
    ## Plot the structure, fields, and figure of merit after each iteration
    # [PART 13]
    print('Finished iteration %d' % (len(fom_list)+1))
    current_fom = -1*am.calc_fom(sim, params)
    fom_list.append(current_fom)

    full_field = sim.field_domains[1]

    Ez, Hx, Hy = sim.saved_fields[1]
    eps = sim.eps.get_values_in(sim.field_domains[1])

    foms = {'Insertion Loss' : fom_list}
    emopt.io.plot_iteration(np.flipud(Ez.real), np.flipud(eps.real), sim.Xreal, sim.Yreal, foms, fname='current_result.pdf', dark=False)

    if(NOT_PARALLEL):
        import matplotlib.pyplot as plt

        extent = full_field.get_bounding_box()[0:4]
        Ez = np.flipud(Ez)

        f = plt.figure()
        ax = f.add_subplot(111)
        im = ax.imshow(Ez.real, extent=extent, vmin=-np.max(Ez.real)/1.0, vmax=np.max(Ez.real)/1.0, cmap='seismic')
        
        ax.set_title('E$_z$', fontsize=18)
        ax.set_xlabel('x [um]', fontsize=14)
        ax.set_ylabel('y [um]', fontsize=14)
        f.colorbar(im)
        f.savefig("gc_pictures/gc_%d.png" % (len(fom_list)+1))
        plt.close(f)

    data = {}
    data['Ez'] = Ez
    data['Hx'] = Hx
    data['Hy'] = Hy
    data['eps'] = eps
    data['params'] = params
    data['foms'] = fom_list

    i = len(fom_list)
    fname = 'data/gc_opt_results'
    emopt.io.save_results(fname, data)

## first of all we define our simulation domain

if __name__ == '__main__':
    wavelength = 1.55 
    W = 28
    H = 8
    dx = 0.02
    dy = dx

    # create the simulation object

    sim = emopt.fdfd.FDFD_TE(W, H, dx, dy, wavelength)

    # Get the actual width and height
    W = sim.X
    H = sim.Y
    M = sim.M
    N = sim.N
    w_pml = sim.w_pml[0] # PML (= perfectly matched layer, absorbs em radiation perfectly, no reflection) width which is the same on all boundaries by default

    # defining the structure 

    n_si = emopt.misc.n_silicon(wavelength)
    eps_si = n_si**2
    eps_clad = 1.444**2

    # the effective indices are precomputed for simplicity.  We can compute
    # these values using emopt.modes, maybe do this
    neff = 2.86
    neff_etched = 2.10
    n0 = np.sqrt(eps_clad)


    ## set up the geometry 

    # set up the initial dimensions of the waveguide structure that we are exciting
    h_wg = 0.28 #height of the waveguide
    h_etch = 0.19 # etch depth
    w_wg_input = 5.0 #how long the initial wg without grating is 
    Ng = 26 #number of grating teeth

    # set the center position of the top silicon and the etches
    y_ts = H/2.0
    y_etch = y_ts + h_wg/2.0 - h_etch/2.0

    # define the starting parameters of the partially-etched grating
    # notably the period and shift between top and bottom layers
    df = 0.8
    theta = 8.0/180.0*pi
    period = wavelength / (df * neff + (1-df)*neff_etched - n0*np.sin(theta))

    # We now build up the grating using a bunch of rectangles
    grating_etch = []

    for i in range(Ng):
        rect_etch = emopt.grid.Rectangle(w_wg_input+i*period, y_etch, (1-df)*period, h_etch)
        rect_etch.layer = 1
        rect_etch.material_value = eps_clad
        grating_etch.append(rect_etch)

    # In general, when optimizing grating couplers an initial duty factor >~60% should be used to ensure that unwanted local minima are avoided. 
    # Notice that the layer of the grating slots is set to 1. 
    # This will be the lowest layer value in our stack to ensure that the grating slot rectangles are on top of the waveguide.

    # grating waveguide --> one big rectangle
    Lwg = Ng*period + w_wg_input
    wg = emopt.grid.Rectangle(Lwg/2.0, y_ts, Lwg, h_wg)
    wg.layer = 2
    wg.material_value = eps_si

    #make cladding and background

    # define substrate big ---> rectangle at the bottom of the simulation
    h_BOX = 2.0
    h_subs = H/2.0 - h_wg/2.0 - h_BOX
    substrate = emopt.grid.Rectangle(W/2.0, h_subs/2.0, W, h_subs)
    substrate.layer = 2
    substrate.material_value = eps_si # silicon

    # set the background material using a rectangle equal in size to the system
    background = emopt.grid.Rectangle(W/2,H/2,W,H)
    background.layer = 3
    background.material_value = eps_clad


    ## combine all the components

    eps = emopt.grid.StructuredMaterial2D(W,H,dx,dy)

    for g in grating_etch:
        eps.add_primitive(g)

    eps.add_primitive(wg)
    eps.add_primitive(substrate)
    eps.add_primitive(background)

    mu = emopt.grid.ConstantMaterial2D(1.0)

    sim.set_materials(eps, mu)

    ## set up sources
    # we will inject power into the grating --> solve for wave modes of the input 
    #                                        --> extract the corresponding current density distribution to create such a mode

    # first make a vertical slice of the sim domain

    w_src= 3.5 #???

    # place the source in the simulation domain
    src_line = emopt.misc.DomainCoordinates(w_pml+2*dx, w_pml+2*dx, H/2-w_src/2, H/2+w_src/2, 0, 0, dx, dy, 1.0)

    # Setup the mode solver.
    mode = emopt.modes.ModeTE(wavelength, eps, mu, src_line, n0=n_si, neigs=4)

    mode.build()
    mode.solve()

    # at this point we have found the modes but we dont know which mode is the
    # one we fundamental mode.  We have a way to determine this, however
    mindex = mode.find_mode_index(0)

    # set the current sources using the mode solver object
    sim.set_sources(mode, src_line, mindex)

    # set up which fields to record
    # --> we want to record the fields in a 1D slice which will be used to calculate the coupling efficiency of the grating coupler
    # --> a second 2D domain which records the fields in the simulation for the purpose of visualization

    mm_line = emopt.misc.DomainCoordinates(w_pml, W-w_pml, H/2.0+2.0, H/2.0+2.0, 0, 0, dx, dy, 1.0)
    full_field = emopt.misc.DomainCoordinates(w_pml, W-w_pml, w_pml, H-w_pml, 0.0, 0.0, dx, dy, 1.0)

    sim.field_domains = [mm_line, full_field]

    sim.build()
    sim.solve_forward()

    
    # check if the gradients are calculated correctly 

    N_coeffs = 5
    design_params = np.zeros(N_coeffs*4+3)
    design_params[0*N_coeffs] = (1-df) * period
    design_params[2*N_coeffs] = period
    design_params[-3] = h_etch
    design_params[-2] = 0.0
    design_params[-1] = h_BOX

    # We initialize our application-specific adjoint method object which is
    # responsible for computing the figure of merit and its gradient with
    # respect to the design parameters of the problem
    am = SiliconGratingAM(sim, grating_etch, wg, substrate, y_ts, w_wg_input, h_wg, H, Ng, N_coeffs, eps_clad, mm_line)

    am.check_gradient(design_params)        

    #set up the optimizer
    fom_list = []
    callback = lambda x : plot_update(x, fom_list, sim, am)

    # setup and run the optimization!
    opt = emopt.optimizer.Optimizer(am, design_params, tol=1e-5, callback_func=callback, opt_method='BFGS', Nmax=40)

    # Run the optimization
    final_fom, final_params = opt.run()