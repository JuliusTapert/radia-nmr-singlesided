import h5py
import numpy as np
from field_sampling import *
from sensitivity import compute_cpmg_signal
from plotting import plot_contour

# ================ Config ================ #

MAT_FILE = "field_maps.mat"
COIL_CURRENT = 4.0

def load_matlab_fields(filepath):
    with h5py.File(filepath, "r") as f:
        B0x = np.array(f["B0x"]).T
        B0z = np.array(f["B0z"]).T
        B1x = np.array(f["B1x"]).T
        B1z = np.array(f["B1z"]).T
        x = np.array(f["x"]).squeeze()
        z = np.array(f["z"]).squeeze()

    # Convert to mm
    x_mm = x * 1e3
    z_mm = z * 1e3

    # Build vector fields (y = 0 plane)
    B0_vec = np.stack([B0x, np.zeros_like(B0x), B0z], axis=-1)
    B1_vec = np.stack([B1x, np.zeros_like(B1x), B1z], axis=-1)

    return B0_vec, B1_vec, x_mm, z_mm

# ================ Spin Dynamics Stuff ================ #

def compute_voxel_size(x_mm, z_mm):
    dx_mm = np.mean(np.diff(x_mm))
    dz_mm = np.mean(np.diff(z_mm))
    return (dx_mm * 1e-3) * (dz_mm * 1e-3)

def solver():
    # Load fields
    B0_vec, B1_vec, x_vals, z_vals = load_matlab_fields(MAT_FILE)

    # You can optionally load in a field from a Python .npz file for testing purposes
    '''B1_data = np.load("B1_shinmagnet.npz")
    B1_vec = B1_data["B"]
    x_vals = B1_data["axis1"]
    z_vals = B1_data["axis2"]'''

    # Magnitudes
    B0_mag = np.linalg.norm(B0_vec, axis=-1)
    B1_mag = np.linalg.norm(B1_vec, axis=-1)

    # Plot B0
    plot_contour(
        B0_mag,
        x_vals,
        z_vals,
        title="|B0| (Matlab)",
        xlabel="X [mm]",
        ylabel="Z [mm]",
        cbar_label="|B0| [T]",
        ncontours=40,
        vmin=0.0,
        vmax=0.2,
    )

    # Compute B1c
    B0m = np.linalg.norm(B0_vec, axis=-1)
    B1dotB0 = np.sum(B1_vec * B0_vec, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        B1c_vec = B1_vec - (B0_vec * B1dotB0[..., None] / B0m[..., None]**2)

    B1c_vec[np.isnan(B1c_vec)] = 0
    B1c_mag = np.linalg.norm(B1c_vec, axis=-1)

    print("Raw B1c mag:",
          np.min(B1c_mag),
          np.max(B1c_mag))

    # Plot B1c
    plot_contour(
        np.abs(B1c_mag),
        x_vals,
        z_vals,
        title="|B1c| (Radia)",
        xlabel="X [mm]",
        ylabel="Z [mm]",
        cbar_label="|B1c| [T]",
        ncontours=50,
        vmin=1e-6,
        vmax=1e-3,
    )

    # Voxel size
    voxel_size = compute_voxel_size(x_vals, z_vals)

    # Signal map
    signalmap = compute_cpmg_signal(
        B0_vec,
        B1_vec,
        horizontal_range=x_vals,
        vertical_range=z_vals,
        I=COIL_CURRENT,
        voxel_size=voxel_size,
    )

    print("Signal stats:")
    print("min:", np.min(np.abs(signalmap)))
    print("max:", np.max(np.abs(signalmap)))
    print("median:", np.median(np.abs(signalmap)))

    # Plot signal
    plot_contour(
        np.abs(signalmap),
        x_vals,
        z_vals,
        title="Sensitivity Map (Matlab B0 and Radia B1)",
        xlabel="X [mm]",
        ylabel="Z [mm]",
        cbar_label="Signal (Normalised)",
        ncontours=100,
        vmin=0,
        vmax=1,
        cmap="plasma",
    )


# =========================
# Entry point
# =========================

def main():
    solver()

if __name__ == "__main__":
    main()