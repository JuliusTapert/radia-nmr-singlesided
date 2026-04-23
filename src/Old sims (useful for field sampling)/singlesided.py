import math
import radia as rad
import radia_vtk as rad_vtk
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from uti_plot import *
from shin_straight_coil import *

#print(sys.path)

#params for inner ring magnet:
    #inner_radius = 1.6
    #outer_radius = 9.5,
    #lx = 11,
    #h = -(25.4/2 - 4.4 - 11/2), (top of inner ring magnet needs to be 4.4 mm below top of outer ring magnet)
    #seg = 40,
    #mg = [0, 0, 1.35]

#params for outer ring magnet:
    #inner_radius = 12.5
    #outer_radius = 25.4
    #lx = 25.4
    #h = 0
    #seg = 40
    #mg = [0, 0, 1.35]

# [inner_radius, outer_radius, lx, h, seg, mg]
tseg = 18
magnet_vertical_spacing = 3.38
outer_magnet_thickness = 25.4
outer_magnet_height = -outer_magnet_thickness/2


# [inner_radius, outer_radius, lx, h, seg, mg]
inner_ring_magnet = [1.6, 9.5, 11, -(11/2+magnet_vertical_spacing), tseg, [0, 0, 1.235]]
outer_ring_magnet = [12.5, 25.4, outer_magnet_thickness, outer_magnet_height, tseg, [0, 0, 1.235]]  
steel_wall = [25.5, 35.4, 12.9, (outer_magnet_height-(12.9/2)), tseg]
steel_baseplate = [2, 35.4, 10, -(25.4 + 5), tseg]

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

        wedge = rad.ObjThckPgn(h, lx, verts, 'z', mg)
        if mg == [0, 0, 0]:
            rad.ObjDrwAtr(wedge, [0.5, 0.5, 0.5])
        else: rad.ObjDrwAtr(wedge, [0.9, 0.2, 0.1])

        ring_wedges.append(wedge)
    return rad.ObjCnt(ring_wedges)

#Contain all objects
steel_assembly = [build_ring_geometry(*steel_baseplate), build_ring_geometry(*steel_wall)]
for i in steel_assembly:
    rad.MatApl(i, steelmat)

full_assembly = rad.ObjCnt([build_ring_geometry(*inner_ring_magnet),        #inner ring magnet
                            build_ring_geometry(*outer_ring_magnet),     #outer ring magnet
                            #+ BuildGeometry(*aluminium_housing)     #aluminium housing ring
                            #*steel_assembly                        #steel baseplate
                            ])  

#Apply symmetries to reduce geometry index
rad.TrfZerPerp(full_assembly, [0,0,0], [1,0,0])
rad.TrfZerPerp(full_assembly, [0,0,0], [0,1,0])

rot = rad.TrfRot([0, 0, 0], [0, 1, 0], math.pi / 2)

def solve_magnetism_xz(fieldobject):
    #This function provides us with an xz-plane view of the magnetic field
    #This helps determine the region of greatest homogeneity
    xMin, xMax, nx = -12.99, 12.99, 201
    zMin, zMax, nz = 0, 20, 201

    y_plane = 0.0
    x_vals = np.linspace(xMin, xMax, nx)
    z_vals = np.linspace(zMin, zMax, nz)

    coords = [[x, y_plane, z]             
              for z in np.linspace(zMin, zMax, nz)
              for x in np.linspace(xMin, xMax, nx)]
    B0_vals = rad.Fld(fieldobject, 'b', coords)
    B0_magnitudes = [(np.linalg.norm(x) * 1e4) for x in B0_vals]
    B0_array = np.array(B0_magnitudes).reshape((nz, nx))


    #Define plot parameters for 1D figures
    ix0 = np.argmin(np.abs(x_vals - 0))
    Bx_cut_y = B0_array[:, ix0]  # vertical cut (x=0)


    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    contour = axs[0].contourf(
        x_vals, z_vals, B0_array,
        levels=np.linspace(1000, 2000, 20), cmap='viridis'
    )
    fig.colorbar(contour, label='|B0| [G]')
    axs[0].set_title('|B0| Contour Map without shield (XZ Plane, Y = 0)')
    axs[0].set_xlabel('X [mm]')
    axs[0].set_ylabel('Z [mm]')

    #1D field map, cut at x = 0
    axs[1].plot(z_vals, Bx_cut_y, color='b')
    axs[1].set_xlim([0, 20])
    #axs[1].set_ylim([1670, 1730])
    axs[1].set_title('|B0| through Z without shield (X = 0)')
    axs[1].set_xlabel('Z [mm from origin]')
    axs[1].set_ylabel('|B0| [G]')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def solve_magnetism_xy(fieldobject):

    #Define plot parameters for 2D figure
    xMin, xMax, nx = -30.99, 30.99, 201
    yMin, yMax, ny = -30.99, 30.99, 201

    z_plane = 4   #Remember when selecting a sampling depth that z-origin is the centre of the outer magnetic ring
    x_vals = np.linspace(xMin, xMax, nx)
    y_vals = np.linspace(yMin, yMax, ny)

    coords = [[x, y, z_plane]
            for y in np.linspace(yMin, yMax, ny)
            for x in np.linspace(xMin, xMax, nx)]

    B0_vals = rad.Fld(fieldobject, 'b', coords)
    B0_magnitudes = [(np.linalg.norm(x) * 1e4) for x in B0_vals]
    B0_array = np.array(B0_magnitudes).reshape((ny, nx))

    #Define plot parameters for 1D figures
    ix0 = np.argmin(np.abs(x_vals - 0))
    iy0 = np.argmin(np.abs(y_vals - 0))

    Bx_cut_y = B0_array[:, ix0]  # vertical cut (x=0)
    Bx_cut_x = B0_array[iy0, :]  # horizontal cut (y=0)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    #2D field map
    im = axs[0].imshow(
        B0_array,
        extent=[xMin, xMax, yMin, yMax],
        origin='lower',
        cmap='viridis',
        aspect='equal'
    )
    axs[0].set_title('|B0| Field Map (Z = %.1f mm)' % (z_plane))
    axs[0].set_xlabel('X [mm]')
    axs[0].set_ylabel('Y [mm]')
    fig.colorbar(im, ax=axs[0], label='|B0| [G]')

    #1D field map, cut at x = 0
    axs[1].plot(y_vals, Bx_cut_y, color='b')
    axs[1].set_title('Vertical Cut (X = 0)')
    axs[1].set_xlabel('Y [mm]')
    axs[1].set_ylabel('|B0| [G]')
    axs[1].grid(True)

    #1D field map, cut at y = 0
    axs[2].plot(x_vals, Bx_cut_x, color = 'g')
    axs[2].set_title('Horizontal Cut (Y = 0)')
    axs[2].set_xlabel('X [mm]')
    axs[2].set_ylabel('|B0| [G]')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

def FieldIsoSurface(fieldobject):

    #Grid params
    xMin, xMax, nx = -30.99, 30.99, 25
    yMin, yMax, ny = -30.99, 30.99, 25
    zMin, zMax, nz = 0, 30.99, 25

    x_vals = np.linspace(xMin, xMax, nx)
    y_vals = np.linspace(yMin, yMax, ny)
    z_vals = np.linspace(zMin, zMax, nz)

    coords = [[x, y, z]
              for z in z_vals
              for y in y_vals
              for x in x_vals]  #x fastest, z slowest
    
    spacing = (
        (xMax - xMin) / (nx - 1),
        (yMax - yMin) / (ny - 1),
        (zMax - zMin) / (nz - 1)
    )

    grid = pv.ImageData(
        dimensions=(nx, ny, nz),
        origin=(xMin, yMin, zMin),
        spacing=spacing
    )

    Bz_vals = np.array(rad.Fld(fieldobject, 'bz', coords))
    Bz_grid = Bz_vals.reshape((nx, ny, nz), order='F')
    grid["Bz"] = Bz_grid.flatten(order="F")
    iso_values = np.linspace(0.1, 0.25, 5)
    contours = grid.contour(isosurfaces=iso_values, scalars="Bz")

    #Plot
    pl = pv.Plotter()
    pl.add_mesh(grid.outline(), color="k")  # optional bounding box
    pl.add_mesh(contours, opacity=0.5, cmap="viridis")
    pl.show()

def GetB0FldList(fieldobject):
    z_list = list(range(3,11))
    z_offset = outer_ring_magnet[2]/2
    
    Bz_vectors = []
    for z in z_list:
        Bz_vectors.append(
            rad.Fld(
                fieldobject,
                'bz',
                [0, 0, (z + z_offset)]
            )
        )

    plt.scatter(z_list, Bz_vectors)
    #plt.ylim(.1775, .181)
    plt.show()

    return Bz_vectors

if __name__=="__main__":

    #Build geometry and display in 3D
    g = full_assembly
    print('Geometry Index', g)
    #rad_vtk.plot_vtk(g)
    #res=rad.Solve(g, 0.00001, 150000)
    #print("Solved: ", res)
    #print(np.array(GetB0FldList(g)) * 1e4)

    solve_magnetism_xz(g)
    #solve_magnetism_xy(g)
    #FieldIsoSurface(g)
