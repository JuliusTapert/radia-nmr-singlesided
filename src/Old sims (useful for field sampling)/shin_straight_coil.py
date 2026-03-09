import math
import radia as rad
import radia_vtk as rad_vtk
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from uti_plot import *

### 
# Shin's paper approximates the coil through the use of straight wires, accounting for differential current flow;
# coilLength = magnetID(1); % [m]; make it the same size as magnet ID
# coilThickness = 2.5e-3; % [m]
# wirePosition = [1.0, 2.5, 4.0, 8.5, 10.0, 11.5] mm
# I = 4 * [+1, +1, +1, -1, -1, -1]; % [A]; coil current at wirePosition
# Q = 20; % probe Q, determining bandwidth
###

lx = 20
x_centre = [1.0, 2.5, 4.0, 8.5, 10.0, 11.5]
wires = []

for idx, x in enumerate(x_centre):
    current = 1 if idx <= 2 else -1 # Note this approximation really only works with these x_centre values. Sorry. If you want more wires, you need to redefine this. Just vibe code if you want
    wires.append(rad.ObjFlmCur([[x,-lx/2,0], [x,lx/2,0]], current))

coil = rad.ObjCnt(wires)
rad.TrfZerPara(coil, [0,0,0], [1,0,0])


def solveMagnetism_xy(fieldobject):

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

def solveMagnetism_yz(fieldobject):
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
        levels=np.linspace(0, 16, 200), 
        cmap='viridis'
    )
    fig.colorbar(contour, label='B1c [G]')
    axs.set_title('B1c Contour Map (YZ Plane, X = 0)')
    axs.set_ylim([0 , 5])
    axs.set_xlabel('Y [mm]')
    axs.set_ylabel('Z [mm]')

    plt.tight_layout()
    plt.show()

def solveMagnetism_xz(fieldobject):
    #This function provides us with an xz-plane view of the magnetic field
    #This helps determine the region of greatest homogeneity
    xMin, xMax, nx = -7.5, 7.5, 201
    zMin, zMax, nz = 0, 8, 201

    y_plane = 0.0
    x_vals = np.linspace(xMin, xMax, nx)
    z_vals = np.linspace(zMin, zMax, nz)

    coords = [[x, y_plane, z]             
              for z in np.linspace(zMin, zMax, nz)
              for x in np.linspace(xMin, xMax, nx)]
    b1_vals = rad.Fld(fieldobject, 'b', coords)
    #b1c_gauss = [((x/2)*10e4) for x in b1x_vals]
    b1_magnitudes = [(np.linalg.norm(x)) for x in b1_vals]
    b1_array = np.array(b1_magnitudes).reshape((nx, nz))
    #b1c_array = np.array(b1c_gauss).reshape((nx,nz))

    #Define plot parameters for 1D figures
    ix0 = np.argmin(np.abs(x_vals))
    Bx_cut_y = b1_array[:, ix0]  # vertical cut (x=0)


    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    contour = axs[0].contourf(
        x_vals, z_vals, b1_array,
        levels=np.linspace(0, 0.2, 200), 
        cmap='viridis'
    )
    fig.colorbar(contour, label='B1 [T]')
    axs[0].set_title('B1 Contour Map (XZ Plane, Y = 0)')
    axs[0].set_ylim([0 , 5])
    axs[0].set_xlabel('X [mm]')
    axs[0].set_ylabel('Z [mm]')

    #1D field map, cut at x = 0
    axs[1].plot(z_vals, Bx_cut_y, color='b')
    axs[1].set_title('B1 through Z (X = 0)')
    axs[1].set_xlabel('Z [mm from origin]')
    axs[1].set_ylabel('B1 [T]')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
    print("Raw B1 mag:",
        np.min(b1_magnitudes),
        np.max(b1_magnitudes))

def exportB1(fieldobject):

    #Define plot parameters for 2D figure
    xMin, xMax, nx = -9.99, 9.99, 201
    yMin, yMax, ny = -9.99, 9.99, 201

    z_plane = 3   #This allows us to visualise the coil more directly
    x_vals = np.linspace(xMin, xMax, nx)
    y_vals = np.linspace(yMin, yMax, ny)

    coords = [[x, y, z_plane]
            for y in np.linspace(yMin, yMax, ny)
            for x in np.linspace(xMin, xMax, nx)]

    b1_vals = rad.Fld(fieldobject, 'b', coords)
    #b1_magntiudes = [(np.linalg.norm(x) * 10e4) for x in b1_vals]
    b1c_vals = [(x/2) * 10e4 for x in b1_vals]
    b1c_array = np.array(b1c_vals).reshape((nx, ny))

if __name__=="__main__":
    g = coil
    #g_inter = rad.RlxPre(g)
    B = rad.Fld(g, 'b', [0,0,3])
    print(B)
    print(np.linalg.norm(B))
    #solveMagnetism_xy(g)
    #solveMagnetism_yz(g)
    solveMagnetism_xz(g)