import radia as rad
import radia_vtk as rad_vtk
from geometry import build_ring_geometry, build_steel_assembly
from coil import build_coil
from field_sampling import *
from plotting import plot_contour
from sensitivity import *
import numpy as np

magnet_specs = {
    "tseg": 18,
    "magnet_vertical_spacing": 4.4,
    "outer_magnet_thickness": 25.4,
    "outer_magnet_height": 25.4/2,
    "mg": [-1.235, 0, 0]
}

coil_positions = [1.0, 2.5, 4.0, 8.5, 10.0, 11.5]
coil_thickness = 2.5
coil_current = 4.0

def magnet_builder():
    # ==========================================================================
    # Building ring-shaped components (inner, outer magnets and wall/baseplate)
    # ==========================================================================

    steelmat = rad.MatSatIsoFrm([20000,2],[0.1,2],[0.1,2])

    class CircularComponent:
        def __init__(self, inner_radius, outer_radius, length, height, segmentation, magnetisation):
            self.data = {"inner_radius": inner_radius, 
                        "outer_radius": outer_radius,
                        "lx": length,
                        "h": height,
                        "tseg": segmentation,
                        "mg": magnetisation}
            
    inner_ring_magnet = CircularComponent(1.6, 
                                    9.5, 
                                    11, 
                                    (11/2+magnet_specs["magnet_vertical_spacing"]), 
                                    magnet_specs["tseg"], 
                                    magnet_specs["mg"])
    outer_ring_magnet = CircularComponent(12.5,
                                    25.4,
                                    magnet_specs["outer_magnet_thickness"],
                                    magnet_specs["outer_magnet_height"],
                                    magnet_specs["tseg"],
                                    magnet_specs["mg"])
    steel_wall = CircularComponent(25.5,
                            35.4,
                            12.9,
                            magnet_specs["outer_magnet_height"]+(12.9/2),
                            magnet_specs["tseg"],
                            [0,0,0])
    steel_baseplate = CircularComponent(2, 
                                    35.4, 
                                    10, 
                                    (25.4 + 5), 
                                    magnet_specs["tseg"], 
                                    [0,0,0])

    # === Build objects ===
    inner_magnet_obj = build_ring_geometry(**inner_ring_magnet.data)
    outer_magnet_obj = build_ring_geometry(**outer_ring_magnet.data)
    steel_assembly = build_steel_assembly(steel_baseplate, steel_wall, steelmat)
    coil_obj = build_coil(coil_positions, coil_thickness, coil_current)

    magnet_assembly = rad.ObjCnt([inner_magnet_obj, 
                                  outer_magnet_obj, 
                                  steel_assembly
                                  ])
    rad.TrfZerPerp(magnet_assembly, [0,0,0], [0,0,1])
    rad.TrfZerPerp(magnet_assembly, [0,0,0], [0,1,0])

    return magnet_assembly, coil_obj
    
def solver(magnet_assembly, coil_obj):

    # === Solve fields ===
    g = rad.TrfMlt(magnet_assembly, rad.TrfRot([0,0,0], [0,1,0], np.pi/2), 1)
    rad_vtk.plot_vtk(g)
    res = rad.Solve(g, 0.00001, 150000)
    print("Geometry index:", g)
    print("Solver result:", res)


    x_sweep = {"min": -10.0, "max": 10.0, "n": 101}
    y_sweep = {"min": -8.0, "max": 8.0, "n": 101}
    z_sweep = {"min": 0, "max": 20.0, "n": 101}

    #These are optionally used in SamplePlane calls as the fixed_coord argument
    x_plane = 0.0
    y_plane = 0.0
    z_plane = 6.0

    
    # === Sample and display B0 === #
    B0_vec, B0_x_vals, B0_z_vals = SamplePlane(
        magnet_assembly,
        plane="xz",
        sweep1=x_sweep, #X values
        sweep2=z_sweep, #Z values
        fixed_coord=y_plane,
    )

    B0_mag = np.linalg.norm(B0_vec, axis=-1)

    """plot_contour(
        B0_mag,
        B0_x_vals,
        B0_z_vals,
        title="|B0| (XZ Plane, Y = 0)",
        xlabel="X [mm]",
        ylabel="Z [mm]",
        cbar_label="|B0| [T]",
        levels=np.linspace(0.1,0.2,20)
    )"""

    # === Sample and display B1c + associated axes just for consistency === #
    B1_vec, B1_x_vals, B1_z_vals = SamplePlane(
        coil_obj,
        plane="xz",
        sweep1=x_sweep,
        sweep2=z_sweep,
        fixed_coord=y_plane,
    )

    B1_mag = np.linalg.norm(B1_vec, axis=-1)
    B1c_mag = np.linalg.norm(compute_B1c(B0_vec,B1_vec),axis=-1)
    print("Raw B1c mag:",
        np.min(B1c_mag),
        np.max(B1c_mag))
    """plot_contour(
        np.abs(B1c_mag),
        B1_x_vals,
        B1_z_vals,
        title="|B1c| (XZ Plane, Y = 0)",
        xlabel="X [mm]",
        ylabel="Z [mm]",
        cbar_label="|B1c| [T]",
        levels=np.linspace(1e-6, 3e-3, 50),
    )"""

    dx_mm = np.mean(np.diff(B1_x_vals))
    dz_mm = np.mean(np.diff(B1_z_vals))
    voxel_size = (dx_mm * 1e-3) * (dz_mm * 1e-3)

    # === Sensitivity map === #
    #Returns a normalised signal map between 0 and 1 that includes both real and complex channels
    signalmap = compute_cpmg_signal(B0_vec, B1_vec, horizontal_range=B1_x_vals, vertical_range=B1_z_vals, I=coil_current, voxel_size=voxel_size)
    print("min:", np.min(np.abs(signalmap)))
    print("max:", np.max(np.abs(signalmap)))
    print("median:", np.median(np.abs(signalmap)))
    plot_contour(
        np.abs(signalmap),
        B1_x_vals,
        B1_z_vals,
        title="Sensitivity Map (XZ Plane, Y = 0)",
        xlabel="X [mm]",
        ylabel="Z [mm]",
        cbar_label="Signal (normalised, dimensionless)",
        levels=np.linspace(0,1,101),
        cmap="plasma"
    )

def main():
    solver(*magnet_builder())

if __name__ == "__main__":
    main()