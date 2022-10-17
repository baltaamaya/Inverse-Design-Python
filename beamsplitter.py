import emopt
from emopt.misc import NOT_PARALLEL, RANK, run_on_master
from emopt.adjoint_method import AdjointMethodPNF2D

import numpy as np
from math import pi

from mpi4py import MPI
class Beamsplitter2D(AdjointMethodPNF2D):
        def __init__(self, sim, mmi, mode_match, line_fom):

                super(Beamsplitter2D, self).__init__(sim, step=5e-7)

                self.mmi = mmi
                self.mode_match = mode_match
                self.line_fom = line_fom


                self.current_fom = 0.0


        def update_system(self, params):
                self.mmi.height = params[0]
                self.mmi.width = params[1]

        @run_on_master
        def calc_f(self, sim, params):
                Ez, Hx, Hy = sim.saved_fields[0]
                self.mode_match.compute(Ez = Ez, Hx=Hx, Hy=Hy)

                fom = -self.mode_match.get_mode_match_forward(1.0)
                self.current_fom = fom

                return fom

        @run_on_master
        def calc_dfdx(self, sim, params):
                dFdEz = np.zeros([sim.M, sim.N], dtype=np.complex128)
                dFdHx = np.zeros([sim.M, sim.N], dtype=np.complex128)
                dFdHy = np.zeros([sim.M, sim.N], dtype=np.complex128)

                # Get the fields which were recorded
                Ez, Hx, Hy = sim.saved_fields[0]
                Psrc = sim.source_power

                self.mode_match.compute(Ez=Ez, Hx=Hx, Hy=Hy)

                j = self.line_fom.j
                k = self.line_fom.k
                dFdEz[j, k] = -self.mode_match.get_dFdEz()
                dFdHx[j, k] = -self.mode_match.get_dFdHx()
                dFdHy[j, k] = -self.mode_match.get_dFdHy()
                return (dFdEz, dFdHx, dFdHy)

        def calc_grad_p(self, sim, params):
                return np.zeros(params.shape)


def plot_update(params, fom_list, sim, am):
        """Plot the current state of the optimization.

        This function is called after each iteration of the optimization
        """
        print('Finished iteration %d.' % (len(fom_list)+1))
        current_fom = -1*am.calc_fom(sim, params)
        fom_list.append(current_fom)


        full_field = sim.field_domains[1]

        Ez, Hx, Hy = sim.saved_fields[1]

        foms = {'Insertion Loss' : fom_list}


        if(NOT_PARALLEL):
                import matplotlib.pyplot as plt

                extent = full_field.get_bounding_box()[0:4]
                Ez = np.flipud(Ez)

                f = plt.figure()
                ax = f.add_subplot(111)
                im = ax.imshow(Ez.real, extent=extent, vmin=-np.max(Ez.real)/1.0, vmax=np.max(Ez.real)/1.0, cmap='seismic')

                # ax.plot(extent[0:2], [sim.Y/2-h_wg/2, sim.Y/2-h_wg/2], 'k-')
                # ax.plot(extent[0:2], [sim.Y/2+h_wg/2, sim.Y/2+h_wg/2], 'k-')

                ax.set_title('E$_z$', fontsize=18)
                ax.set_xlabel('x [um]', fontsize=14)
                ax.set_ylabel('y [um]', fontsize=14)
                f.colorbar(im)
                f.savefig("beamsplitter_pictures/beamsplitter_%d.png" % (len(fom_list)))
                plt.close(f)

if __name__ == '__main__':

####################################################################################
# Simulation parameters
####################################################################################

        X = 7.0
        Y = 5.0
        dx = 0.02
        dy = dx

        wavelength = 1.55

        #create sim object
        sim = emopt.fdfd.FDFD_TE(X,Y,dx,dy,wavelength)

        w_pml = dx * 15
        sim.w_pml = [w_pml, w_pml,w_pml, w_pml]

        #get real parameters
        X = sim.X
        Y = sim.Y
        M = sim.M
        N = sim.N

        # defining the structures
        w_wg = 0.45
        L_in = X/2+1
        L_out = X/2+1
        L_mmi = 3.2
        w_mmi = 2.5
        offset_out1 = (Y-w_mmi)/2 + w_wg/2 + 0.25
        offset_out2 = Y - offset_out1

        n_si = emopt.misc.n_silicon(wavelength)
        eps_si = n_si**2
        eps_clad = 1.444**2

        wg_in = emopt.grid.Rectangle(X/4, Y/2, L_in, w_wg)
        wg_in.layer = 1
        wg_in.material_value = eps_si

        mmi = emopt.grid.Rectangle(X/2, Y/2, L_mmi, w_mmi)
        mmi.layer = 1
        mmi.material_value = eps_si


        wg_out = emopt.grid.Rectangle(3*X/4, offset_out1, L_out, w_wg)
        wg_out.layer = 1
        wg_out.material_value = eps_si

        wg_out2 = emopt.grid.Rectangle(3*X/4, offset_out2, L_out, w_wg)
        wg_out2.layer = 1
        wg_out2.material_value = eps_si

        rbg = emopt.grid.Rectangle(X/2, 0, 2*X, 2*Y)
        rbg.layer = 2
        rbg.material_value = eps_clad

        #combine all components
        eps = emopt.grid.StructuredMaterial2D(X,Y,dx,dy)
        eps.add_primitive(wg_in)
        eps.add_primitive(mmi)
        eps.add_primitive(wg_out)
        eps.add_primitive(wg_out2)
        eps.add_primitive(rbg)

        mu = emopt.grid.ConstantMaterial2D(1.0)

        sim.set_materials(eps, mu)
        sim.build()

        # setup sources
        w_src = 1

        src_line = emopt.misc.DomainCoordinates(w_pml+2*dx, w_pml+2*dx, Y/2 - w_src/2, Y/2 + w_src/2, 0, 0, dx, dy, 1.0)

        # Setup the mode solver.
        mode = emopt.modes.ModeTE(wavelength, eps, mu, src_line, n0=n_si, neigs=4)
        
        mode.build()
        mode.solve()

        #select the fundamental mode 
        mindex = mode.find_mode_index(0)

        # set the current sources using the mode solver object
        sim.set_sources(mode, src_line, mindex)

        # # setting up fom mode
        line_fom = emopt.misc.DomainCoordinates(X - w_pml - 2*dx, X - w_pml - 2*dx, offset_out1 - w_src/2 , offset_out1 + w_src/2, 0, 0, dx, dy, 1.0)
        fom_mode = emopt.modes.ModeTE(wavelength, eps, mu, line_fom, n0=n_si, neigs=4)

        fom_mode.build()
        fom_mode.solve()

        # # mode match
        mode_match = None
        if(NOT_PARALLEL):         
                Ezm = fom_mode.get_field_interp(0, 'Ez')
                Hxm = fom_mode.get_field_interp(0, 'Hx')
                Hym = fom_mode.get_field_interp(0, 'Hy')

                mode_match = emopt.fomutils.ModeMatch([1,0,0], sim.dy, Ezm=Ezm, Hxm=Hxm, Hym=Hym)

        full_field = emopt.misc.DomainCoordinates(0, X-0, 0, Y-0, 0.0, 0.0, dx, dy, 1.0)


        sim.field_domains = [line_fom, full_field]

        # sim.build()
        # sim.solve_forward()


        #setup the optimization

        design_params = np.array([w_mmi, L_mmi])

        # The adjoint method class for this problem is defined above!
        am = Beamsplitter2D(sim, mmi, mode_match, line_fom)

        #check gradients!
        am.check_gradient(design_params)

        #set up optimizer
        fom_list = []
        callback = lambda x : plot_update(x, fom_list, sim, am)

        # setup and run the optimization!
        opt = emopt.optimizer.Optimizer(am, design_params, tol=1e-5, callback_func=callback, opt_method='BFGS', Nmax=40)

        # Run the optimization
        final_fom, final_params = opt.run()

# sim_area = emopt.misc.DomainCoordinates(0.12, X-0.12, 0.12, Y-0.12, 0.0, 0.0, dx, dy, 1.0)
# Ez = sim.get_field_interp('Ez', sim_area)

# if(NOT_PARALLEL):
#         import matplotlib.pyplot as plt

#         extent = sim_area.get_bounding_box()[0:4]

#         f = plt.figure()
#         ax = f.add_subplot(111)
#         im = ax.imshow(Ez.real, extent=extent,
#                             vmin=-np.max(Ez.real)/1.0,
#                             vmax=np.max(Ez.real)/1.0,
#                             cmap='seismic')
#         f.colorbar(im)
#         ax.set_title('E$_z$', fontsize=18)
#         ax.set_xlabel('x [um]', fontsize=14)
#         ax.set_ylabel('y [um]', fontsize=14)
#         f.savefig('BeamsplitterUnOPT.png')
