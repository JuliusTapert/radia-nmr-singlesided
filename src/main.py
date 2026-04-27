# dependencies
import radia as rad
import matplotlib.pyplot as plt
import radia_vtk as rad_vtk
import plotly.graph_objects as go
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# modules
from geometry import build_ring_geometry, build_steel_assembly
from coil import build_coil
from field_sampling import *
from plotting import plot_contour
from sensitivity import *


# =========================================================
# Data Models
# =========================================================

@dataclass
class CircularComponent:
    inner_radius: float
    outer_radius: float
    lz: float
    h: float
    tseg: int
    mg: List[float]

    def as_dict(self):
        return vars(self)


@dataclass
class CoilConfig:
    positions: List[float]
    thickness: float
    current: float
    lx: float = 15.0


@dataclass
class MagnetPreset:
    name: str
    inner_magnet: CircularComponent
    outer_magnet: CircularComponent
    coil: CoilConfig
    recess: float
    wall: Optional[CircularComponent] = None
    baseplate: Optional[CircularComponent] = None


# =========================================================
# Preset Registry
# =========================================================

PRESETS: Dict[str, MagnetPreset] = {
    "SN001": MagnetPreset(
        name="SN001",
        recess=4.35,
        inner_magnet=CircularComponent(
            inner_radius=1.6,
            outer_radius=9.5,
            lz=11,
            h=-(11 / 2 + 4.35),
            tseg=18,
            mg=[0, 0, 1.235],
        ),
        outer_magnet=CircularComponent(
            inner_radius=12.5,
            outer_radius=25.4,
            lz=25.4,
            h=-25.4 / 2,
            tseg=18,
            mg=[0, 0, 1.235],
        ),
        coil=CoilConfig(
            positions=[0.5, 1.5, 2.5, 6, 7, 8],
            thickness=1.4,
            current=4.0,
        ),
        wall=CircularComponent(
            inner_radius=25.5,
            outer_radius=35.4,
            lz=12.9,
            h=(-25.4 / 2) - (12.9 / 2),
            tseg=18,
            mg=[0, 0, 0],
        ),
        baseplate=CircularComponent(
            inner_radius=2,
            outer_radius=35.4,
            lz=10,
            h=-(25.4 + 5),
            tseg=18,
            mg=[0, 0, 0],
        ),
    ),

    "Shin": MagnetPreset(
        name="Shin",
        recess=3.38,
        inner_magnet=CircularComponent(
            inner_radius=1.5875,
            outer_radius=9.525,
            lz=11.1125,
            h=-(11.1125 / 2 + 3.38) - 0.05,
            tseg=18,
            mg=[0, 0, 1.3],
        ),
        outer_magnet=CircularComponent(
            inner_radius=12.7,
            outer_radius=25.4,
            lz=25.4,
            h=-25.4 / 2,
            tseg=18,
            mg=[0, 0, 1.3],
        ),
        coil=CoilConfig(
            positions=[1.0, 2.5, 4.0, 8.5, 10.0, 11.5],
            thickness=2.5,
            current=4.0,
        ),
    ),

    "Test": MagnetPreset(
        name="Test",
        recess=4.35,
        inner_magnet=CircularComponent(
            inner_radius=1.6,
            outer_radius=9.5,
            lz=11,
            h=-(11 / 2 + 4.35),
            tseg=18,
            mg=[0, 0, 1.235],
        ),
        outer_magnet=CircularComponent(
            inner_radius=12.5,
            outer_radius=25.4,
            lz=25.4,
            h=-25.4 / 2,
            tseg=18,
            mg=[0, 0, 1.235],
        ),
        coil=CoilConfig(
            positions=[0.5, 1.5, 2.5, 6, 7, 8],
            thickness=1.4,
            current=4.0,
        ),
        wall=CircularComponent(
            inner_radius=25.5,
            outer_radius=35.4,
            lz=12.9,
            h=(-25.4 / 2) - (12.9 / 2),
            tseg=18,
            mg=[0, 0, 0],
        ),
        baseplate=CircularComponent(
            inner_radius=2,
            outer_radius=35.4,
            lz=10,
            h=-(25.4 + 5),
            tseg=18,
            mg=[0, 0, 0],
        ),
    )
}


# =========================================================
# Builder
# =========================================================

def magnet_builder(preset_name: str):
    preset = PRESETS[preset_name]

    steelmat = rad.MatSatIsoFrm([20000, 2], [0.1, 2], [0.1, 2])

    inner_obj = build_ring_geometry(**preset.inner_magnet.as_dict())
    outer_obj = build_ring_geometry(**preset.outer_magnet.as_dict())

    components = [inner_obj, outer_obj]

    if preset.wall:
        steel_obj = build_steel_assembly(preset.baseplate, preset.wall, steelmat)
        components.append(steel_obj)
        needs_solve = True

    magnet_assembly = rad.ObjCnt(components)

    rad.TrfZerPerp(magnet_assembly, [0, 0, 0], [1, 0, 0])
    rad.TrfZerPerp(magnet_assembly, [0, 0, 0], [0, 1, 0])

    coil_obj = build_coil(
        preset.coil.positions,
        preset.coil.thickness,
        preset.coil.current,
        lx=preset.coil.lx,
    )

    return magnet_assembly, coil_obj, needs_solve, preset


# =========================================================
# Solver
# =========================================================

def solver(g, coil, needs_solve, preset: MagnetPreset):
    if needs_solve:
        rad.Solve(g, 1e-5, 150000)

    print("Geometry index:", g)

    x_sweep = {"min": -10.0, "max": 10.0, "n": 201}
    z_sweep = {"min": 3.0, "max": 20.0, "n": 201}
    y_plane = 0.0

    # -----------------------------
    # B0 Sampling
    # -----------------------------
    B0_vec, B0_x_vals, B0_z_vals = sample_plane(
        g,
        plane="xz",
        sweep1=x_sweep,
        sweep2=z_sweep,
        fixed_coord=y_plane,
    )

    B0_mag = np.linalg.norm(B0_vec, axis=-1)

    x0_idx = np.argmin(np.abs(B0_x_vals - 0.0))
    B0_z_profile = B0_mag[:, x0_idx]

    # -----------------------------
    # B1 Sampling
    # -----------------------------
    B1_vec, B1_x_vals, B1_z_vals = sample_plane(
        coil,
        plane="xz",
        sweep1=x_sweep,
        sweep2=z_sweep,
        fixed_coord=y_plane,
    )

    export_plane(B1_vec, B1_x_vals, B1_z_vals, f"B1_{preset.name}")

    B1_mag = np.linalg.norm(B1_vec, axis=-1)
    B1c_mag = np.linalg.norm(compute_B1c(B0_vec, B1_vec), axis=-1)

    print("Raw B1c mag:", np.min(B1c_mag), np.max(B1c_mag))

    """plot_contour(
        np.abs(B1c_mag),
        B1_x_vals,
        B1_z_vals,
        title="|B1c| (XZ Plane, Y = 0)",
        xlabel="X [mm]",
        ylabel="Z [mm]",
        cbar_label="|B1c| [T]",
        vmin=1e-6,
        vmax=1e-3,
    )"""

    B1_z_profile = B1_mag[:, x0_idx]
    B1c_z_profile = B1c_mag[:, x0_idx]

    # -----------------------------
    # Sensitivity Map
    # -----------------------------
    dx_mm = np.mean(np.diff(B1_x_vals))
    dz_mm = np.mean(np.diff(B1_z_vals))
    voxel_size = (dx_mm * 1e-3) * (dz_mm * 1e-3)

    signalmap = compute_cpmg_signal(
        B0_vec,
        B1_vec,
        horizontal_range=B1_x_vals,
        vertical_range=B1_z_vals,
        I=preset.coil.current,
        voxel_size=voxel_size,
    )

    signal_z_profile = np.abs(signalmap[:, x0_idx])

    print("Signal stats:")
    print("min:", np.min(np.abs(signalmap)))
    print("max:", np.max(np.abs(signalmap)))
    print("median:", np.median(np.abs(signalmap)))

    """plot_contour(
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
    )"""

    # -----------------------------
    # Combined Plot
    # -----------------------------
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=B0_z_vals,
        y=B0_z_profile,
        mode="lines",
        name="|B0|",
        yaxis="y1",
    ))

    fig.add_trace(go.Scatter(
        x=B0_z_vals,
        y=B1c_z_profile,
        mode="lines",
        name="|B1c|",
        yaxis="y2",
    ))

    fig.add_trace(go.Scatter(
        x=B0_z_vals,
        y=signal_z_profile,
        mode="lines",
        name="|Signal|",
        yaxis="y3",
    ))

    fig.update_layout(
        title=f"{preset.name}: B0 / B1c / Signal along Z",
        xaxis=dict(title="Z [mm]", range=[0, 10]),
        yaxis=dict(title="|B0| [T]", side="left"),
        yaxis2=dict(title="|B1c| [T]", overlaying="y", side="right"),
        yaxis3=dict(title="|Signal|", overlaying="y", side="right", position=.9),
    )

    fig.show()


# =========================================================
# Main
# =========================================================

def main():
    mag, coil, needs_solve, preset = magnet_builder("Test")
    solver(mag, coil, needs_solve, preset)

if __name__ == "__main__":
    main()