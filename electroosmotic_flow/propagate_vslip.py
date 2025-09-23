import numpy as np
import cupy as cp
import spreadinterp
import libMobility as lm
import functools
from typing import List, Optional
from scipy.sparse.linalg import gmres, LinearOperator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmap import Colormap

GOLD = (255/255, 200/255, 1/255)
GOLD = (1,1,1)

def inverseMdot(
    v: cp.ndarray,
    mdot: callable,
    x0: Optional[cp.ndarray] = None,
    callback: Optional[callable] = None,
    atol: Optional[float] = 1e-3,
    rtol: Optional[float] = 1e-3,
) -> cp.ndarray:
    '''
    Solves the linear system A·x = v using the GMRES iterative method,
    where A is implicitly represented by the function `mdot`.

    Parameters:
    v (cp.ndarray): The right-hand side of the linear system.
    mdot (callable): A function that applies the linear operator A to a vector.
    x0 (Optional[cp.ndarray]): Initial guess for the solution. Defaults to None.
    callback (Optional[callable]): Optional callback called at each iteration. Defaults to None.
    atol (Optional[float]): Absolute tolerance for convergence. Defaults to 1e-3.
    rtol (Optional[float]): Relative tolerance for convergence. Defaults to 1e-3.

    Returns:
    cp.ndarray: Solution vector x such that A·x ≈ v.
    '''
    n = v.size
    op = LinearOperator((n, n), matvec=mdot)
    x, exitCode = gmres(op, v, x0=x0, callback=callback, atol=atol, rtol=rtol, callback_type='legacy')
    if exitCode != 0:
        raise ValueError("gmres did not converge")
    return x


def create_electrode_particles(
        v_slip: cp.array,
        hydrodynamicRadius: float,
        n_repeats: int,
        kernel: dict,
        Lx0: Optional[float] = 190,
        Lz0: Optional[float] = 160,
        Ny: Optional[int] = 24,
):
    '''
    Generates particle positions and velocities for an electrode system by interpolating
    the slip velocity field and replicating the system along the x-axis.

    Parameters:
    v_slip (cp.array): Slip velocity values in the x-direction.
    hydrodynamicRadius (float): Hydrodynamic radius of particles.
    n_repeats (int): Number of times to replicate the system along the x-axis.
    kernel (dict): Spreadinterp kernel used for interpolation.
    Lx0 (Optional[float]): Initial domain length in x. Defaults to 190.
    Lz0 (Optional[float]): Initial domain length in z. Defaults to 160.
    Ny (Optional[int]): Number of grid points in y. Defaults to 12.

    Returns:
    Tuple[List[float], cp.array, cp.array]: 
        - Domain size after replication [Lx, Ly, Lz],
        - Particle positions (N x 3),
        - Particle velocities (N x 3)
    '''
    field =  cp.zeros((v_slip.size, Ny, 3), dtype=cp.float32)
    field[:,:,0] = cp.tile(v_slip, (Ny, 1)).T
    field = cp.ascontiguousarray(cp.array(field[:, :, cp.newaxis, :]))

    L = [Lx0, 0.0, Lz0]
    L[1] = L[0]*np.shape(field)[1]/np.shape(field)[0]

    x = cp.arange(-L[0]/2, L[0]/2, 2*hydrodynamicRadius) + hydrodynamicRadius
    y = cp.arange(-L[1]/2, L[1]/2, 2*hydrodynamicRadius) + hydrodynamicRadius
    z = cp.arange(0, L[2], 2*hydrodynamicRadius) + hydrodynamicRadius

    x_electrode, y_electrode = cp.meshgrid(x, y, indexing='ij')

    numberParticles = len(x_electrode.flatten())
    particle_positions = cp.zeros((numberParticles, 3), dtype=cp.float32)
    particle_positions[:,0] = x_electrode.flatten()
    particle_positions[:,1] = y_electrode.flatten()

    particle_velocities = spreadinterp.interpolate(
            particle_positions, field, [L[0], L[1], 0.0], kernel=kernel
        )


    # Create copies of the system in the x direction
    pos_repeated = cp.zeros((n_repeats*numberParticles, 3), dtype=cp.float32)
    pos_repeated[:,0] = cp.array([particle_positions[:,0] + (i+0.5)*L[0] for i in range(n_repeats)]).flatten()
    pos_repeated[:,1] = cp.tile(particle_positions[:,1], n_repeats)
    pos_repeated[:,2] = cp.tile(particle_positions[:,2], n_repeats)

    vel_repeated = cp.zeros_like(pos_repeated)
    vel_repeated[:,0] = cp.array([particle_velocities[:,0] for i in range(n_repeats)]).flatten()
    vel_repeated[:,1] = cp.tile(particle_velocities[:,1], n_repeats)
    vel_repeated[:,2] = cp.tile(particle_velocities[:,2], n_repeats)

    # Rescale the system to the desired length
    L[0] *= n_repeats
    pos_repeated[:,0] = pos_repeated[:,0] - L[0]/2 # Center the system at x = 0

    # Remove particles with zero velocity in the x direction to accelerate the simulation
    pos = pos_repeated[np.abs(vel_repeated[:,0])>0]
    vel = vel_repeated[np.abs(vel_repeated[:,0])>0]

    return L, pos, vel

def create_wall_particles(
        hydrodynamicRadius: float,
        L: List
):
    '''
    Creates wall particles with zero velocity at x = L[0]/2 across all y and z.

    Parameters:
    hydrodynamicRadius (float): Hydrodynamic radius of wall particles.
    L (List): Domain dimensions [Lx, Ly, Lz].

    Returns:
    Tuple[cp.array, cp.array]: 
        - Wall particle positions (N x 3),
        - Wall particle velocities (zeros) (N x 3)
    '''
    y = cp.arange(-L[1]/2, L[1]/2, 2*hydrodynamicRadius) + hydrodynamicRadius
    z = cp.arange(0, L[2], 2*hydrodynamicRadius) + hydrodynamicRadius
    y_wall, z_wall = cp.meshgrid(y, z, indexing='ij')
    wall_positions = cp.zeros((len(y_wall.flatten()), 3), dtype=cp.float32)
    wall_positions[:,0] = L[0]/2
    wall_positions[:,1] = y_wall.flatten()
    wall_positions[:,2] = z_wall.flatten()
    wall_velocities = cp.zeros_like(wall_positions)

    return wall_positions, wall_velocities

def init_solver(L: List, hydrodynamicRadius: float):
    '''
    Initializes a libmobility DPStokes solver with periodic boundary conditions
    in x and y and wall boundaries in z.

    Parameters:
    L (List): Domain dimensions [Lx, Ly, Lz].
    hydrodynamicRadius (float): Hydrodynamic radius of particles.

    Returns:
    lm.DPStokes: Configured DPStokes solver object.
    '''
    periodicityXY = 'periodic'
    periodicityZ = 'two_walls'

    wallHeight = -2*hydrodynamicRadius
    solver = lm.DPStokes(periodicityX=periodicityXY, periodicityY=periodicityXY, periodicityZ=periodicityZ)
    solver.setParameters(Lx=L[0],Ly=L[1], zmin=wallHeight, zmax=L[2]+2*hydrodynamicRadius)
    solver.initialize(
            viscosity=0.01/(6*np.pi),
            hydrodynamicRadius=hydrodynamicRadius,
            includeAngular=False,
        )
    return solver

def solve_inverse_problem(solver: lm.DPStokes,
                        pos: cp.array,
                        vel: cp.array,
                        rtol: Optional[float] = 1e-3,
                        atol: Optional[float] = 0):
    '''
    Solves the inverse Stokes problem using GMRES to determine the particle forces
    that produce the observed velocities.

    Parameters:
    solver (lm.DPStokes): Initialized libmobility DPStokes solver.
    pos (cp.array): Particle positions (N x 3).
    vel (cp.array): Particle velocities (N x 3).
    rtol (Optional[float]): Relative tolerance for GMRES. Defaults to 1e-3.
    atol (Optional[float]): Absolute tolerance for GMRES. Defaults to 0.

    Returns:
    cp.array: Computed particle forces (N x 3).
    '''
    solver.setPositions(pos.get())

    def mdot(solver, vector):
        mf, _ = solver.Mdot(forces=vector)
        return mf

    mdot_partial = functools.partial(mdot, solver)
    forces = inverseMdot(
        vel.get().flatten(),
        mdot_partial,
        rtol=rtol,
        atol=atol
    )
    forces = forces.reshape(-1,3)
    return forces

def create_tracer_particles(Lx_min, Lx_max, Lz_min, Lz_max, Nx, Nz):
    '''
    Creates tracer particles arranged on a 2D lattice (x-z plane) at y=0 
    to sample fluid velocity.

    Parameters:
    Lx_min (float): Minimum x-coordinate.
    Lx_max (float): Maximum x-coordinate.
    Lz_min (float): Minimum z-coordinate.
    Lz_max (float): Maximum z-coordinate.
    Nx (int): Number of tracers along x.
    Nz (int): Number of tracers along z.

    Returns:
    cp.array: Tracer particle positions (Nx * Nz x 3).
    '''
    x = cp.linspace(Lx_min, Lx_max, Nx)
    y = cp.array([0])
    z = cp.linspace(Lz_min, Lz_max, Nz)
    xx_trac ,yy_trac ,zz_trac = cp.meshgrid(x,y,z,indexing='ij')
    tracer_pos = cp.zeros([Nx*Nz,3])
    tracer_pos[:,0] = xx_trac.flatten()
    tracer_pos[:,1] = yy_trac.flatten()
    tracer_pos[:,2] = zz_trac.flatten()
    return tracer_pos

def sample_bulk_velocities(solver: lm.DPStokes,
                           pos: cp.array,
                           forces: cp.array,
                           Lx_min: float,
                           Lx_max: float,
                           Lz_min: float,
                           Lz_max: float,
                           Nx: int):
    '''
    Samples the fluid velocity field at tracer points by computing M·F using the solver.

    Parameters:
    solver (lm.DPStokes): Initialized DPStokes solver.
    pos (cp.array): Particle positions (N x 3).
    forces (cp.array): Particle forces (N x 3).
    Lx_min (float): Minimum x for tracer grid.
    Lx_max (float): Maximum x for tracer grid.
    Lz_min (float): Minimum z for tracer grid.
    Lz_max (float): Maximum z for tracer grid.
    Nx (int): Number of tracers along x-direction.

    Returns:
    Tuple[np.ndarray, np.ndarray]: 
        - Grid positions of tracers [Nx, 1, Nz, 3],
        - Corresponding velocities [Nx, 1, Nz, 3]
    '''
    Nz = int(Nx/(Lx_max - Lx_min)*(Lz_max - Lz_min)) # To sample a square lattice
    tracer_pos = create_tracer_particles(Lx_min, Lx_max, Lz_min, Lz_max, Nx, Nz)
    tracer_forces = tracer_pos*0
    total_pos = cp.concatenate([pos,tracer_pos],axis=0)
    total_force = cp.concatenate([cp.array(forces),tracer_forces],axis=0)
    solver.setPositions(total_pos)
    total_vel, _ = solver.Mdot(forces=total_force)
    grid_pos = np.reshape(total_pos[len(pos):,:], (Nx, 1, Nz, 3)) #Not taking into account the boundary blobs
    grid_vel = np.reshape(total_vel[len(pos):,:], (Nx, 1, Nz, 3)) #Not taking into account the boundary blobs
    return grid_pos, grid_vel

def plot_arrows(strm, ax):
    '''
    Plots arrows along the streamlines to indicate flow direction.
    Parameters:
    strm: Streamline object from matplotlib's streamplot.
    ax: Matplotlib axis to plot on.
    Returns:
    None
    '''

    for line in strm.lines.get_segments():
        x, y = line[:,0], line[:,1]

        indices = np.linspace(0, len(x)-1, 17, dtype=int)
        # por ejemplo cada 20 puntos
        if np.abs(x[indices].max()-x[indices].min())>100 or np.abs(y[indices].max()-y[indices].min())>40:  # Verifica si la línea percola
            x_point = 10000
            y_point = 10000
            for i in indices[:-2]:
                if (x[i]-x_point)**2+(y[i]-y_point)**2 > 30**2:  # Cada 30 unidades de distancia
                    x_point = x[i]
                    y_point = y[i]
                    if (x[i+1]-x[i])**2+(y[i+1]-y[i])**2 > 1e-5:  # Evita flechas en líneas muy cortas
                        ax.arrow(x[i], y[i], (x[i+1]-x[i]), (y[i+1]-y[i]), head_width=4, head_length=4, fc=GOLD, ec=GOLD)
                    elif (x[i+2]-x[i])**2+(y[i+2]-y[i])**2 > 1e-5:  # Evita flechas en líneas muy cortas
                        ax.arrow(x[i], y[i], (x[i+2]-x[i]), (y[i+2]-y[i]), head_width=4, head_length=4, fc=GOLD, ec=GOLD)
                    else:
                        continue
        else:
            i = indices[10]
            if (x[i+2]-x[i])**2+(y[i+2]-y[i])**2 > 1e-5:  # Evita flechas en líneas muy cortas
                ax.arrow(x[i], y[i], (x[i+2]-x[i]), (y[i+2]-y[i]), head_width=4, head_length=4, fc=GOLD, ec=GOLD)

# Plotting the streamlines

# Función para truncar colormap
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plotter(grid_pos, grid_vel,colorbar=True):
    '''
    Generates a streamline plot and a velocity profile comparison against
    experimental data.

    Parameters:
    grid_pos (np.ndarray): Grid positions of tracers [Nx, Ny, Nz, 3].
    grid_vel (np.ndarray): Tracer velocities [Nx, Ny, Nz, 3].

    Returns:
    Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]: 
        - Streamline plot figure,
        - Velocity profile figure.
    '''
    #Change the font to computer modern
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'serif'

    u = grid_vel[:,0,:,0].get().T
    v = grid_vel[:,0,:,2].get().T
    color = np.sqrt(u*u+v*v)
    max_color = 300
    color[color > max_color] = max_color  # Set a maximums
    streamPlot, ax = plt.subplots(figsize=(12, 6))
    x_min = grid_pos[:,:,:,0].get().min()
    x_max = grid_pos[:,:,:,0].get().max()
    z_min = grid_pos[:,:,:,2].get().min()
    z_max = grid_pos[:,:,:,2].get().max()
    cmap = truncate_colormap(Colormap('seaborn:mako').to_mpl(), 0.1, 0.9)
    ax.imshow(color, extent=(x_min, x_max, z_min, z_max), origin="lower", cmap=cmap)
    # Add cbar for the imshow
    if colorbar==True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%",pad=0.05)    
        cbar = plt.colorbar(ax.images[0], cax=cax)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel(r'$v(\mu \text{m/s})$', fontsize=16)

    # Streamplot
    strm = ax.streamplot(grid_pos[:,0,:,0].get().T,
                  grid_pos[:,0,:,2].get().T,
                  u, v,
                  density=0.5,
                  linewidth=2, arrowsize=0,
                  broken_streamlines=True,
                  color=GOLD)
    plot_arrows(strm, ax)

    fontsize = 20
    ax.set_ylim((0, 190))
    ax.set_xlim((x_min, x_max))
    ax.set_aspect('equal')
    ax.set_ylabel(r'$z(\mu \text{m})$',fontsize=fontsize)
    ax.set_xlabel(r'$x(\mu \text{m})$',fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)

    # Plotting the velocity profile
    z = grid_pos[30,0,:,2].flatten().get()
    z_min = 0 # Minimum z value to consider in the profile
    mask = z>z_min
    mask2 = ((grid_pos[:,:,:,2].flatten().get())>z_min)

    # Experimental data from P. Garcia-Sanchez et al.
    x_exp = [44.21052631578947, 59.57894736842105, 81.26315789473684, 97.26315789473684, 127.57894736842104, 143.1578947368421, 179.15789473684208, 190.52631578947367]
    y_exp = [-49.855072463768124, -27.08827404479578, 23.504611330698253, 57.65480895915675, 68.61660079051381, 66.08695652173913, 26.034255599473, -0.5270092226614054]

    u_mean = np.mean(u[:,:],axis=1)

    vProfile = plt.figure()
    plt.axhline(0, color='k', linestyle='--', linewidth=3)
    plt.plot(z, u_mean, '-', label='Libmobility',linewidth=3)
    plt.plot(z, -223.64 + 4.5377*z - 0.01762*z*z, 'r--', linewidth=3,label='Fit from P. Garcia-Sanchez et al.')

    plt.plot(x_exp, y_exp, 'ko', label='Exp. data from P. Garcia-Sanchez et al.', markersize=7)
    plt.xlim([0, 200])
    plt.ylim([-240, 80])
    plt.xlabel(r'$z(\mu \text{m})$', fontsize=16)
    plt.ylabel(r'$v(\mu \text{m/s})$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)
    return streamPlot, vProfile

def libmobility_electroosmotic_flow(
        v_slip: cp.array,
        hydrodynamicRadius: float,
        n_repeats: int,
        Lx0: float,
        Lz0: float,
        Lx_min_trac: float,
        Lx_max_trac: float,
        Lz_min_trac: float,
        Lz_max_trac: float,
        Nx_trac: int,
        kernel: dict,
        Ny: Optional[int] = 12
        ):
    '''
    Simulates electroosmotic flow using the libmobility library and generates
    streamline and velocity profile plots to compare with P. Garcia-Sanchez results.

    Parameters:
    v_slip (cp.array): Slip velocities in the x-direction.
    hydrodynamicRadius (float): Hydrodynamic radius of particles.
    n_repeats (int): Number of times to replicate the particle system along x.
    Lx0 (float): Initial domain size in x.
    Lz0 (float): Initial domain size in z.
    Lx_min_trac (float): Minimum x for tracer region.
    Lx_max_trac (float): Maximum x for tracer region.
    Lz_min_trac (float): Minimum z for tracer region.
    Lz_max_trac (float): Maximum z for tracer region.
    Nx_trac (int): Number of tracer points along x.
    kernel (dict): Spreadinterp kernel for interpolation.
    Ny (Optional[int]): Grid size in y-direction (affects Ly). Defaults to 12.

    Returns:
    Tuple[np.ndarray, np.ndarray]
        - Grid positions (i, j, k, dim)
        - Grid velocities (i, j, k, dim)
         Details:
        - i, j, k: indices along the grid in the x, y, and z directions, respectively.
        - dim (components):
            - dim = 0 → x-coordinate
            - dim = 1 → y-coordinate
            - dim = 2 → z-coordinate
    '''


    L, pos, vel = create_electrode_particles(v_slip=v_slip,
                                            hydrodynamicRadius=hydrodynamicRadius,
                                            n_repeats=n_repeats,
                                            kernel=kernel,
                                            Lx0=Lx0,
                                            Lz0=Lz0,
                                            Ny=Ny)

    wall_positions, wall_velocities = create_wall_particles(hydrodynamicRadius=hydrodynamicRadius, L=L)

    # Combine wall and particle positions and velocities
    pos = cp.concatenate((pos, wall_positions), axis=0)
    vel = cp.concatenate((vel, wall_velocities), axis=0)

    solver = init_solver(L=L, hydrodynamicRadius=hydrodynamicRadius)
    forces = solve_inverse_problem(solver, pos, vel)

    grid_pos, grid_vel = sample_bulk_velocities(solver, pos, forces, Lx_min_trac, Lx_max_trac, Lz_min_trac, Lz_max_trac, Nx_trac)
    return grid_pos, grid_vel
    #solver.clean()

if __name__ == '__main__':
    # Vslip is given in m/s with Vrms = 1V and Lambda = 1
    # P. Garcia-Sanchez uses Vpp = 6V so Vrms = 6/sqrt(2) V
    # They state that Lambda = 0.03 to mach the experimental data
    # We will use the same values for our simulations

    Lambda = 0.03
    Vrms = (6 / np.sqrt(2))  # Vrms in Volts
    um2m = 1e6  # Conversion factor from m/s to um/s
    v_slip = cp.load('Vslip.npy')[1:-1,0]*um2m*Vrms**2*Lambda

    Lx0 = 160  # um
    Lz0 = 190  # um
    hydrodynamicRadius = 1 # um This is the radius of the particles determines how many particles are in the simulation

    # Kernel of interpolation can be changed, using the one that gives the best results
    kernel = spreadinterp.create_kernel('gaussian', width=4*hydrodynamicRadius, cutoff=12*hydrodynamicRadius)
    #kernel = spreadinterp.create_kernel('peskin3pt')

    #Number of copies of the system in the x direction. [Number of electrodes = 4*n_repeats]
    n_repeats = 20

    Lx_min_trac = -Lx0
    Lx_max_trac = Lx0
    Nx_trac = 1000
    Lz_min_trac = 0
    Lz_max_trac = Lz0

    grid_pos, grid_vel = libmobility_electroosmotic_flow(
        v_slip=v_slip,
        hydrodynamicRadius=hydrodynamicRadius,
        n_repeats=n_repeats,
        Lx0=Lx0,
        Lz0=Lz0,
        kernel=kernel,
        Lx_min_trac=Lx_min_trac,
        Lx_max_trac=Lx_max_trac,
        Lz_min_trac=Lz_min_trac,
        Lz_max_trac=Lz_max_trac,
        Nx_trac=Nx_trac
        )
    
    sp, vp = plotter(grid_pos=grid_pos, grid_vel=grid_vel, colorbar=False)
    vp.savefig("velocityProfile.svg", format='svg') # Uncomment to save the figure
    sp.savefig("transientStreamplot.svg", format='svg') # Uncomment to save the figure

    #Number of copies of the system in the x direction. [Number of electrodes = 4*n_repeats]
    n_repeats = 20

    Lx_min_trac = (n_repeats-4)*Lx0/2
    Lx_max_trac = n_repeats*Lx0/2
    Nx_trac = 1000
    Lz_min_trac = 0
    Lz_max_trac = Lz0

    grid_pos, grid_vel = libmobility_electroosmotic_flow(
        v_slip=v_slip,
        hydrodynamicRadius=hydrodynamicRadius,
        n_repeats=n_repeats,
        Lx0=Lx0,
        Lz0=Lz0,
        kernel=kernel,
        Lx_min_trac=Lx_min_trac,
        Lx_max_trac=Lx_max_trac,
        Lz_min_trac=Lz_min_trac,
        Lz_max_trac=Lz_max_trac,
        Nx_trac=Nx_trac
        )
    
    sp, vp = plotter(grid_pos=grid_pos, grid_vel=grid_vel, colorbar=False)
    vp.savefig("velocityProfileWall.svg", format='svg') # Uncomment to save the figure
    sp.savefig("transientStreamplotWall.svg", format='svg') # Uncomment to save the figure


