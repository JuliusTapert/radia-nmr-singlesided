import math
import radia as rad
import radia_vtk as rad_vtk
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from uti_plot import *

tseg = 8

inner_ring_magnet = [1.6, 9.5, 11, 0, tseg, [-1, 0, 0]]
outer_ring_magnet = [12.5, 25.4, 25.4, 0, tseg, [-1, 0, 0]]
magnet_list = [inner_ring_magnet, outer_ring_magnet]

def BuildRingGeometry(inner_radius, outer_radius, lx, h, seg, mg = [0, 0, 0]):
    #create arbitrarily angular trapezoid based on segmentation
    theta_segment =  math.pi / (2 * seg)
    ring_wedges = []
    
    for i in range(seg):
        phi0 = i * theta_segment
        phi1 = (i+1) * theta_segment

        #precess vertices around unit circle (outer then inner in reverse)
        verts = [
            [outer_radius * math.cos(phi0), outer_radius * math.sin(phi0)],
            [outer_radius * math.cos(phi1), outer_radius * math.sin(phi1)],
            [inner_radius * math.cos(phi1), inner_radius * math.sin(phi1)],
            [inner_radius * math.cos(phi0), inner_radius * math.sin(phi0)]
        ]

        wedge = rad.ObjThckPgn(h, lx, verts, mg)
        if mg == [0, 0, 0]:
            rad.ObjDrwAtr(wedge, [0.5, 0.5, 0.5])
        else: rad.ObjDrwAtr(wedge, [0.9, 0.2, 0.1])

        ring_wedges.append(wedge)
    return rad.ObjCnt(ring_wedges)

def SolveMagnetism_xz(fieldobject, i):
    #This function provides us with an xz-plane view of the magnetic field
    #This helps determine the region of greatest homogeneity
    xMin, xMax, nx = -12.99, 12.99, 201
    zMin, zMax, nz = (i[2]/2 + 1), 30.99, 201

    y_plane = 0.0
    x_vals = np.linspace(xMin, xMax, nx)
    z_vals = np.linspace(zMin, zMax, nz)

    coords = [[x, y_plane, z]             
              for z in np.linspace(zMin, zMax, nz)
              for x in np.linspace(xMin, xMax, nx)]
    Bz_vals = rad.Fld(fieldobject, 'bz', coords)
    Bz_array = np.array(Bz_vals).reshape((nz, nx))

    #Define plot parameters for 1D figures
    ix0 = np.argmin(np.abs(x_vals - 0))
    Bx_cut_y = Bz_array[:, ix0]  # vertical cut (x=0)


    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    contour = axs[0].contourf(
        x_vals, z_vals, Bz_array,
        levels=50, cmap='viridis'
    )
    fig.colorbar(contour, label='Bz [T]')
    axs[0].set_title('Bz Contour Map (XZ Plane, Y = 0)')
    axs[0].set_xlabel('X [mm]')
    axs[0].set_ylabel('Z [mm]')

    #1D field map, cut at x = 0
    axs[1].plot(z_vals, Bx_cut_y, color='b')
    axs[1].set_title('Bz through Z (X = 0)')
    axs[1].set_xlabel('Z [mm]')
    axs[1].set_ylabel('Bz [T]')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    for i in magnet_list:
        full_assembly = BuildRingGeometry(*i)

        #Apply rotation and add symmetries to reduce geometry index 
        rot = rad.TrfRot([0, 0, 0], [0, 1, 0], math.pi / 2)
        g = rad.TrfMlt(full_assembly, rot, 1)
        rad.TrfZerPerp(g, [0,0,0], [1,0,0])
        rad.TrfZerPerp(g, [0,0,0], [0,1,0])
        print('Geometry Index', g)
        

        #rad_vtk.plot_vtk(g)
        SolveMagnetism_xz(g, i)