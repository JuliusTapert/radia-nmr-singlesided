import radia as rad
import radia_vtk as rad_vtk
from geometry import build_ring_geometry, build_steel_assembly
from coil import build_coil
from field_sampling import *
from plotting import plot_contour
from sensitivity import compute_cpmg_signal
import numpy as np

def main():
    # === User-tweakable parameters ===
    tseg = 18
    magnet_vertical_spacing = 4.4
    outer_magnet_thickness = 25.4
    outer_magnet_height = outer_magnet_thickness/2

    inner_ring_magnet = [1.6, 9.5, 11, (11/2+magnet_vertical_spacing), tseg, [-1.235, 0, 0]]
    outer_ring_magnet = [12.5, 25.4, outer_magnet_thickness, outer_magnet_height, tseg, [-1.235, 0, 0]]  
    steel_wall = [25.5, 35.4, 12.9, (outer_magnet_height+(12.9/2)), tseg]
    steel_baseplate = [2, 35.4, 10, (25.4 + 5), tseg]

    steelmat = rad.MatSatIsoFrm([20000,2],[0.1,2],[0.1,2])
    coil_positions = [1.0, 2.5, 4.0, 8.5, 10.0, 11.5]

    x_sweep = {"min": -8.0, "max": 8.0, "n": 101}
    y_sweep = {"min": -8.0, "max": 8.0, "n": 101}
    z_sweep = {"min": 0.0, "max": 20.0, "n": 101}
    x_plane = 0.0
    y_plane = 0.0
    z_plane = 6.0

    # === Build objects ===
    steel_assembly = build_steel_assembly(steel_baseplate, steel_wall, steelmat)
    inner_magnet_obj = build_ring_geometry(*inner_ring_magnet)
    outer_magnet_obj = build_ring_geometry(*outer_ring_magnet)
    coil_obj = build_coil(coil_positions)

    magnet_assembly = rad.ObjCnt([inner_magnet_obj, outer_magnet_obj, steel_assembly])
    rad.TrfZerPerp(magnet_assembly, [0,0,0], [0,0,1])
    rad.TrfZerPerp(magnet_assembly, [0,0,0], [0,1,0])

    # === Solve fields and export ===
    g = rad.TrfMlt(magnet_assembly, rad.TrfRot([0,0,0], [0,1,0], np.pi/2), 1)
    #rad_vtk.plot_vtk(g)
    res = rad.Solve(g, 0.00001, 150000)
    print("Geometry index:", g)
    print("Solver result:", res)

    # === Sample and display B0 === #
    B0_vec, B0_x_vals, B0_z_vals = sample_plane(
        magnet_assembly,
        plane="xz",
        sweep1={"min": -8.0, "max": 8.0, "n": 101}, #X values
        sweep2={"min": 0.0, "max": 20.0, "n": 101}, #Z values
        fixed_coord=y_plane,
    )

    B0_mag = np.linalg.norm(B0_vec, axis=2)

    plot_contour(
        B0_mag,
        B0_x_vals,
        B0_z_vals,
        title="|B0| (XZ Plane, Y = 0)",
        xlabel="X [mm]",
        ylabel="Z [mm]",
        cbar_label="|B0| [T]",
        levels=np.linspace(0.1,0.2,20)
    )

    # === Sample and display B1 === #
    B1_vec, B1_x_vals, B1_z_vals = sample_plane(
        coil_obj,
        plane="xz",
        sweep1={"min": -8.0, "max": 8.0, "n": 101},
        sweep2={"min": 0.0, "max": 5.00, "n": 101},
        fixed_coord=y_plane,
    )

    B1_mag = np.linalg.norm(B1_vec, axis=2)

    plot_contour(
        B1_mag,
        B1_x_vals,
        B1_z_vals,
        title="|B1| (XZ Plane, Y = 0)",
        xlabel="X [mm]",
        ylabel="Z [mm]",
        cbar_label="|B1| [T]",
        levels=np.linspace(0, 12e-4, 101)

    )

    # === Sensitivity map === #
    signalmap = compute_cpmg_signal(B0_vec, B1_vec)
    plot_contour(
        signalmap,
        B0_x_vals,
        B0_z_vals,
        title="Sensitivity Map (XZ Plane, Y = 0)",
        xlabel="X [mm]",
        ylabel="Z [mm]",
        cbar_label="Sensitivity [a.u.]",
        levels=np.linspace(-1,1,101),
        cmap="plasma"
    )


if __name__ == "__main__":
    main()