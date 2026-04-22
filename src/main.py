import radia as rad
import matplotlib.pyplot as plt
import radia_vtk as rad_vtk
import plotly.graph_objects as go
from geometry import build_ring_geometry, build_steel_assembly
from coil import build_coil
from field_sampling import *
from plotting import plot_contour
from sensitivity import *
import numpy as np

magnet_specs = {
    "tseg": 18,
    "magnet_recess": 4.35,     # SN001 recess = 4.4 mm
    "outer_magnet_thickness": 25.4,
    "outer_magnet_height": -25.4/2,
    "mg": [0, 0, 1.235]
}

magnet_specs_shin = {
    "tseg": 18,
    "magnet_recess": 3.38,
    "outer_magnet_thickness": 25.4,
    "outer_magnet_height": -25.4/2,
    "mg": [0,0,1.3]
}

coil_positions_shin = [1.0, 2.5, 4.0, 8.5, 10.0, 11.5]
coil_positions_sn001 = [0.5, 1.5, 2.5, 6, 7, 8]
coil_thickness = 2.5                   # 1.4mm is based on the .pcb file included with the manufacturer specifications 
coil_current = 4.0

class CircularComponent:
    def __init__(self, inner_radius, outer_radius, length, height, segmentation, magnetisation):
        self.data = {"inner_radius": inner_radius, 
                    "outer_radius": outer_radius,
                    "lz": length,
                    "h": height,
                    "tseg": segmentation,
                    "mg": magnetisation}
        
inner_ring_magnet = CircularComponent(1.6, 
                                9.5, 
                                11, 
                                -(11/2+magnet_specs["magnet_recess"]), 
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
                                magnet_specs["outer_magnet_height"]-(12.9/2),
                                magnet_specs["tseg"],
                                [0,0,0])
steel_baseplate = CircularComponent(2, 
                                35.4, 
                                10, 
                                -(25.4 + 5), 
                                magnet_specs["tseg"], 
                                [0,0,0])
shin_inner_magnet = CircularComponent(1.5875,
                                        9.525,
                                        11.1125,
                                        -(11.1125/2 + magnet_specs_shin["magnet_recess"])-0.05,
                                        magnet_specs_shin["tseg"],
                                        magnet_specs_shin["mg"])
shin_outer_magnet = CircularComponent(12.7,
                                        25.4,
                                        25.4,
                                        -25.4/2,
                                        magnet_specs_shin["tseg"],
                                        magnet_specs_shin["mg"])

def magnet_builder(inner, outer, steel=True, coil="SN001"):
    # ==========================================================================
    # Building ring-shaped components (inner, outer magnets and wall/baseplate)
    # ==========================================================================

    steelmat = rad.MatSatIsoFrm([20000,2],[0.1,2],[0.1,2])

    # === Build objects ===
    inner_magnet_obj = build_ring_geometry(**inner.data)
    outer_magnet_obj = build_ring_geometry(**outer.data)
    steel_assembly = build_steel_assembly(steel_baseplate, steel_wall, steelmat)

    components = [inner_magnet_obj, outer_magnet_obj]
    if steel:
        components.append(steel_assembly)
    magnet_assembly = rad.ObjCnt(components)

    rad.TrfZerPerp(magnet_assembly, [0,0,0], [1,0,0])
    rad.TrfZerPerp(magnet_assembly, [0,0,0], [0,1,0])

    if coil == "SN001":
        coil_obj = build_coil(coil_positions_sn001, 1.4, coil_current, lx=15.00)
    elif coil == "Shin":
        coil_obj = build_coil(coil_positions_shin, 2.5, coil_current, lx=15.00)
    
    return magnet_assembly, coil_obj
    
def solver(g, coil):
    # === Solve fields ===
    #res = rad.Solve(g, 1e-5, 150000)
    print("Geometry index:", g)

    x_sweep = {"min": -10.0, "max": 10.0, "n": 201}
    z_sweep = {"min": 3.0, "max": 20.0, "n": 201}
    y_plane = 0.0

    # === Sample B0 === #
    B0_vec, B0_x_vals, B0_z_vals = sample_plane(
        g,
        plane="xz",
        sweep1=x_sweep,
        sweep2=z_sweep,
        fixed_coord=y_plane,
    )
    #export_plane(B0_vec, B0_x_vals, B0_z_vals, "B0_shinmagnet")

    B0_mag = np.linalg.norm(B0_vec, axis=-1)

    # --- 1D B0 plot ---
    x0_idx = np.argmin(np.abs(B0_x_vals - 0.0))
    B0_z_profile = B0_mag[:, x0_idx]


    # --- 2D B0 contour ---
    """plot_contour(
        B0_mag,
        B0_x_vals,
        B0_z_vals,
        title="|B0| (Radia) (XZ Plane, Y = 0)",
        xlabel="X [mm]",
        ylabel="Z [mm]",
        cbar_label="|B0| [T]",
        vmin=0.1,
        vmax=0.2
        )"""

    # === Sample B1 === #
    B1_vec, B1_x_vals, B1_z_vals = sample_plane(
        coil,
        plane="xz",
        sweep1=x_sweep,
        sweep2=z_sweep,
        fixed_coord=y_plane,
    )

    export_plane(B1_vec, B1_x_vals, B1_z_vals, "B1_shinmagnet")
    B1_mag = np.linalg.norm(B1_vec, axis=-1)
    B1c_mag = np.linalg.norm(compute_B1c(B0_vec, B1_vec), axis=-1)

    print("Raw B1c mag:",
          np.min(B1c_mag),
          np.max(B1c_mag))

    # --- 2D B1c contour ---
    plot_contour(
        np.abs(B1_mag),
        B1_x_vals,
        B1_z_vals,
        title="|B1| (XZ Plane, Y = 0)",  # FIXED LABEL
        xlabel="X [mm]",
        ylabel="Z [mm]",
        cbar_label="|B1| [T]",
        vmin=1e-6,
        vmax=1e-3,
    )

    # --- 1D B1c plot ---
    x0_idx = np.argmin(np.abs(B1_x_vals - 0.0))
    B1_z_profile = B1_mag[:, x0_idx]
    B1c_z_profile = B1c_mag[:, x0_idx]

    """fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=B1_z_vals,
        y=B1c_z_profile,
        mode='lines',
        name='|B1c|',
        hovertemplate="Z=%{x:.2f} mm<br>|B1c|=%{y:.3e} T<extra></extra>"
    ))
    fig.update_xaxes(range=[0, 10], title="Z [mm]")
    fig.update_yaxes(title="|B1c| [T]")
    fig.update_layout(title="B1c along Z at X = 0")
    fig.show()"""

    # === Sensitivity map === #
    dx_mm = np.mean(np.diff(B1_x_vals))
    dz_mm = np.mean(np.diff(B1_z_vals))
    voxel_size = (dx_mm * 1e-3) * (dz_mm * 1e-3)

    signalmap = compute_cpmg_signal(
        B0_vec,
        B1_vec,
        horizontal_range=B1_x_vals,
        vertical_range=B1_z_vals,
        I=coil_current,
        voxel_size=voxel_size
    )
    signal_z_profile = np.abs(signalmap[:, x0_idx])

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
        cbar_label="Signal (normalised)",
        cmap="plasma",
        ncontours=101,
        vmin=0,
        vmax=1,
    )

    # 1D plots of B0, B1, and signal
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=B0_z_vals,
        y=B0_z_profile,
        mode='lines',
        name='|B0|',
        yaxis='y1',
        hovertemplate="Z=%{x:.2f} mm<br>|B0|=%{y:.3e} T<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=B0_z_vals,
        y=B1c_z_profile,
        mode='lines',
        name='|B1c|',
        yaxis='y2',
        hovertemplate="Z=%{x:.2f} mm<br>|B1|=%{y:.3e} T<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=B0_z_vals,
        y=signal_z_profile,   # you must define this like B0/B1c
        mode='lines',
        name='|Signal|',
        yaxis='y3',
        hovertemplate="Z=%{x:.2f} mm<br>|Signal|=%{y:.3e}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 10], title="Z [mm]")
    fig.update_yaxes(title="|B0| [T]")

    fig.update_layout(
        title="B0 and B1c along Z at X = 0; Recess = " + str(magnet_specs["magnet_recess"]),
        xaxis=dict(title="Z [mm]", range=[0, 10]),
        yaxis=dict(
            title="|B0| [T]",
            range=[0.15, 0.16],
            side='left'
        ),
        yaxis2=dict(
            title="|B1| [T]",
            overlaying='y',
            side='right'
        ),
        yaxis3=dict(
            title="|Signal|",
            overlaying='y',
            side='right',
            position=0.8
        ),
    )
    fig.show()


def main():
    mag, coil = magnet_builder(shin_inner_magnet, shin_outer_magnet, steel=False, coil="Shin")
    solver(mag, coil)

if __name__ == "__main__":
    main()