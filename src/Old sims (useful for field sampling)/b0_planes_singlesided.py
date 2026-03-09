import math
import radia as rad
import radia_vtk as rad_vtk
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from uti_plot import *

tseg = 18
magnet_vertical_spacing = 4.4
outer_magnet_thickness = 25.4
outer_magnet_height = outer_magnet_thickness/2

# [inner_radius, outer_radius, lx, h, seg, mg]
inner_ring_magnet = [1.6, 9.5, 11, (11/2+magnet_vertical_spacing), tseg, [-1.235, 0, 0]]
outer_ring_magnet = [12.5, 25.4, outer_magnet_thickness, outer_magnet_height, tseg, [-1.235, 0, 0]]  
steel_wall = [25.5, 35.4, 12.9, (outer_magnet_height+(12.9/2)), tseg]
steel_baseplate = [2, 35.4, 10, (25.4 + 5), tseg]

steelmat = rad.MatSatIsoFrm([20000,2],[0.1,2],[0.1,2])


def build_ring_geometry(inner_radius, outer_radius, lx, h, seg, mg = [0, 0, 0]):
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

#Contain all objects
steel_assembly = [build_ring_geometry(*steel_baseplate), build_ring_geometry(*steel_wall)]
for i in range(len(steel_assembly)):
    rad.MatApl(steel_assembly[i], steelmat)
    i += 1

full_assembly = rad.ObjCnt([build_ring_geometry(*inner_ring_magnet),        #inner ring magnet
                            build_ring_geometry(*outer_ring_magnet),     #outer ring magnet
                            #+ BuildGeometry(*aluminium_housing)     #aluminium housing ring
                            *steel_assembly])  #steel baseplate

#Apply symmetries to reduce geometry index
rad.TrfZerPerp(full_assembly, [0,0,0], [0,0,1])
rad.TrfZerPerp(full_assembly, [0,0,0], [0,1,0])

rot = rad.TrfRot([0, 0, 0], [0, 1, 0], math.pi / 2)

def solve_magnetism_xz(fieldobject):
    #This function provides us with an xz-plane view of the magnetic field
    #This helps determine the region of greatest homogeneity
    xMin, xMax, nx = -10, 10, 100
    zMin, zMax, nz = 0, 20, 100

    y_plane = 0.0
    x_vals = np.linspace(xMin, xMax, nx)
    z_vals = np.linspace(zMin, zMax, nz)

    coords = [[x, y_plane, z]             
              for z in np.linspace(zMin, zMax, nz)
              for x in np.linspace(xMin, xMax, nx)]
    B0_vals = rad.Fld(fieldobject, 'b', coords)
    B0_magnitudes = [(np.linalg.norm(x) * 1e4) for x in B0_vals]
    B0_array = np.array(B0_magnitudes).reshape((nz, nx))

    fig, axs = plt.subplots(1, 1, figsize = (6, 6))
    contour = axs.contourf(
        x_vals, z_vals, B0_array,
        levels=np.linspace(1000, 2000), cmap='viridis'
    )
    fig.colorbar(contour, label='|B0| [G]')
    axs.set_title('|B0| Contour Map (XZ Plane, Y = 0)')
    axs.set_xlabel('X [mm]')
    axs.set_ylabel('Z [mm]')
    plt.tight_layout()
    plt.show()
    return B0_vals

def solve_magnetism_yz(fieldobject):
    #This function provides us with an xz-plane view of the magnetic field
    #This helps determine the region of greatest homogeneity
    yMin, yMax, ny = -10, 10, 100
    zMin, zMax, nz = 0, 20, 100

    x_plane = 0.0
    y_vals = np.linspace(yMin, yMax, ny)
    z_vals = np.linspace(zMin, zMax, nz)

    coords = [[x_plane, y, z]             
              for z in np.linspace(zMin, zMax, nz)
              for y in np.linspace(yMin, yMax, ny)]
    B0_vals = rad.Fld(fieldobject, 'b', coords)
    B0_magnitudes = [(np.linalg.norm(i) * 1e4) for i in B0_vals]
    B0_array = np.array(B0_magnitudes).reshape((nz, ny))

    fig, axs = plt.subplots(1, 1, figsize = (6, 6))
    contour = axs.contourf(
        y_vals, z_vals, B0_array,
        levels=np.linspace(1000, 2000), cmap='viridis'
    )
    fig.colorbar(contour, label='|B0| [G]')
    axs.set_title('|B0| Contour Map (YZ Plane, X = 0)')
    axs.set_xlabel('Y [mm]')
    axs.set_ylabel('Z [mm]')
    plt.tight_layout()
    plt.show()

def solve_magnetism_xy(fieldobject):

    #Define plot parameters for 2D figure
    xMin, xMax, nx = -8, 8, 101
    yMin, yMax, ny = -8, 8, 101

    z_plane = 6   #Remember when selecting a sampling depth that z-origin is the centre of the outer magnetic ring
    x_vals = np.linspace(xMin, xMax, nx)
    y_vals = np.linspace(yMin, yMax, ny)

    coords = [[x, y, z_plane]
            for y in np.linspace(yMin, yMax, ny)
            for x in np.linspace(xMin, xMax, nx)]

    B0_vals = rad.Fld(fieldobject, 'b', coords)
    B0_magnitudes = [(np.linalg.norm(x) * 1e4) for x in B0_vals]
    B0_array = np.array(B0_magnitudes).reshape((ny, nx))

    fig, axs = plt.subplots(1, 1, figsize=(6, 6))

    contour = axs.contourf(
        x_vals, y_vals, B0_array,
        levels=np.linspace(1200, 2000), cmap='viridis'
    )
    axs.set_title('|B0| Field Map (Z = %.1f mm)' % (z_plane))
    axs.set_xlabel('X [mm]')
    axs.set_ylabel('Y [mm]')
    fig.colorbar(contour, ax=axs, label='|B0| [G]')

    plt.tight_layout()
    plt.show()
    return 



if __name__=="__main__":

    g = rad.TrfMlt(full_assembly, rot, 1)
    res=rad.Solve(g, 0.00001, 150000)
    print(res) 

    print(solve_magnetism_xz(g))
    solve_magnetism_yz(g)
    solve_magnetism_xy(g)