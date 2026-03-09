import math
import radia as rad
import radia_vtk as rad_vtk
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import pandas as pd
from uti_plot import *

current = 1

#These are a set of points between which a straight line will be drawn to define the coil filament
coil = rad.ObjFlmCur([[-0.5,-7.5,0], [-0.5,7.5,0], [-0.845,7.845,0], [-2.00,7.845,0], [-3.00,7.45,0], [-3.80,7,0], [-5.40,5.90,0], [-6.7,4.30,0], [-7.40,2.73,0], [-8.00,1,0], [-8.00,-1,0], [-7.40,-2.73,0], [-6.7,-4.3,0], [-5.4,-5.9,0], [-3.80,-7,0], [-2.00,-7.51,0], [-1.50,-7.00,0], 
                      [-1.50,6.00,0], [-2.00,6.50,0], [-2.84,6.42,0], [-3.92,5.75,0], [-4.82,5.10,0], [-5.60,4.30,0], [-6.31,3.00,0], [-6.75,1.90,0], [-6.93,0.5,0], [-6.93,-0.5,0], [-6.75,-1.90,0], [-6.31,-3.00,0], [-5.60,-4.30,0], [-4.82,-5.10,0], [-3.92,-5.75,0], [-3.00,-5.64,0], [-2.50,-5.15,0],
                      [-2.50,4.25,0], [-2.95,4.70,0], [-3.72,4.70,0], [-4.30,4.20,0], [-5.00,3.40,0], [-5.42,2.50,0], [-5.84,2.00,0], [-5.95,0.50,0], [-5.95,-0.50,0], [-5.84,-2.00,0], [-5.42,-2.50,0], [-5.00,-3.40,0], [-4.30,-4.20,0], [-3.72,-4.70,0]
                      ], current)

#create a planar symmetry parallel to the x axis
rad.TrfZerPara(coil, [0,0,0], [1,0,0])


def solve_magnetism_xy(fieldobject):

    #Define plot parameters for 2D figure
    xMin, xMax, nx = -9.99, 9.99, 201
    yMin, yMax, ny = -9.99, 9.99, 201

    z_plane = 3   #This allows us to visualise the coil more directly
    x_vals = np.linspace(xMin, xMax, nx)
    y_vals = np.linspace(yMin, yMax, ny)

    coords = [[x, y, z_plane]
            for y in np.linspace(yMin, yMax, ny)
            for x in np.linspace(xMin, xMax, nx)]

    b1_vals = rad.Fld(fieldobject, 'bx', coords)
    #b1_magntiudes = [(np.linalg.norm(x) * 10e4) for x in b1_vals]
    b1c_vals = [(x/2) * 10e4 for x in b1_vals]
    b1c_array = np.array(b1c_vals).reshape((nx, ny))
    

    #Define plot parameters for 1D figures
    ix0 = np.argmin(np.abs(x_vals - 0))
    iy0 = np.argmin(np.abs(y_vals - 0))

    Bx_cut_y = b1c_array[:, ix0]  # vertical cut (x=0)
    Bx_cut_x = b1c_array[iy0, :]  # horizontal cut (y=0)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    #2D field map
    im = axs[0].imshow(
        b1c_array,
        extent=[xMin, xMax, yMin, yMax],
        origin='lower',
        cmap='viridis',
        aspect='equal'
    )
    axs[0].set_title('B1c Field Map (Z = %.1f mm)' % (z_plane))
    axs[0].set_xlabel('X [mm]')
    axs[0].set_ylabel('Y [mm]')
    fig.colorbar(im, ax=axs[0], label='B1c [G]')

    #1D field map, cut at x = 0
    axs[1].plot(y_vals, Bx_cut_y, color='b')
    axs[1].set_title('Vertical Cut (X = 0)')
    axs[1].set_xlabel('Y [mm]')
    axs[1].set_ylabel('B1c [G]')
    axs[1].grid(True)

    #1D field map, cut at y = 0
    axs[2].plot(x_vals, Bx_cut_x, color = 'g')
    axs[2].set_title('Horizontal Cut (Y = 0)')
    axs[2].set_xlabel('X [mm]')
    axs[2].set_ylabel('B1c [G]')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

def solve_magnetism_yz(fieldobject):
    #This function provides us with an xz-plane view of the magnetic field
    #This helps determine the region of greatest homogeneity
    yMin, yMax, ny = -9.99, 9.99, 201
    zMin, zMax, nz = 0, 9.99, 201

    x_plane = 0.0
    y_vals = np.linspace(yMin, yMax, ny)
    z_vals = np.linspace(zMin, zMax, nz)

    coords = [[x_plane, y, z]             
              for z in np.linspace(zMin, zMax, nz)
              for y in np.linspace(yMin, yMax, ny)]
    b1x_vals = rad.Fld(fieldobject, 'bx', coords)
    b1c_gauss = [((i/2)*10e4) for i in b1x_vals]
    b1c_array = np.array(b1c_gauss).reshape((ny,nz))

    fig, axs = plt.subplots(1, 1, figsize = (6, 6))
    contour = axs.contourf(
        y_vals, z_vals, b1c_array,
        levels=np.linspace(0, 30, 200), 
        cmap='viridis'
    )
    fig.colorbar(contour, label='B1c [G]')
    axs.set_title('B1c Contour Map (YZ Plane, X = 0)')
    axs.set_ylim([0 , 5])
    axs.set_xlabel('Y [mm]')
    axs.set_ylabel('Z [mm]')

    plt.tight_layout()
    plt.show()

def solve_magnetism_xz(fieldobject):
    #This function provides us with an xz-plane view of the magnetic field
    #This helps determine the region of greatest homogeneity
    xMin, xMax, nx = -9.99, 9.99, 201
    zMin, zMax, nz = 0, 9.99, 201

    y_plane = 0.0
    x_vals = np.linspace(xMin, xMax, nx)
    z_vals = np.linspace(zMin, zMax, nz)

    coords = [[x, y_plane, z]             
              for z in np.linspace(zMin, zMax, nz)
              for x in np.linspace(xMin, xMax, nx)]
    b1x_vals = rad.Fld(fieldobject, 'bx', coords)
    b1c_gauss = [((x/2)*10e4) for x in b1x_vals]
    #b1_magntiudes = [(np.linalg.norm(x) * 10e4) for x in b1_vals]
    #b1_array = np.array(b1_magntiudes).reshape((nx, nz))
    b1c_array = np.array(b1c_gauss).reshape((nx,nz))

    #Define plot parameters for 1D figures
    ix0 = np.argmin(np.abs(x_vals))
    Bx_cut_y = b1c_array[:, ix0]  # vertical cut (x=0)


    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    contour = axs[0].contourf(
        x_vals, z_vals, b1c_array,
        levels=np.linspace(-30, 30, 200), 
        cmap='viridis'
    )
    fig.colorbar(contour, label='B1c [G]')
    axs[0].set_title('B1c Contour Map (XZ Plane, Y = 0)')
    axs[0].set_ylim([0 , 5])
    axs[0].set_xlabel('X [mm]')
    axs[0].set_ylabel('Z [mm]')

    #1D field map, cut at x = 0
    axs[1].plot(z_vals, Bx_cut_y, color='b')
    axs[1].set_title('B1c through Z (X = 0)')
    axs[1].set_xlabel('Z [mm from origin]')
    axs[1].set_ylabel('B1c [G]')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    g = rad.ObjCnt([coil])
    B = rad.Fld(g, 'b', [0,0,0])
    print(np.linalg.norm(B))
    #solve_magnetism_xy(g)
    #solve_magnetism_yz(g)
    solve_magnetism_xz(g)
