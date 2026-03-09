import numpy as np
import radia as rad


def SamplePlane(field_object, plane, sweep1, sweep2, fixed_coord=0.0):

    n1 = sweep1["n"]
    n2 = sweep2["n"]

    axis1_vals = np.linspace(sweep1["min"], sweep1["max"], n1)
    axis2_vals = np.linspace(sweep2["min"], sweep2["max"], n2)

    coords = []

    if plane == "xy":
        z_plane = fixed_coord
        for y in axis2_vals:
            for x in axis1_vals:
                coords.append([x, y, z_plane])

    elif plane == "xz":
        y_plane = fixed_coord
        for z in axis2_vals:
            for x in axis1_vals:
                coords.append([x, y_plane, z])

    elif plane == "yz":
        x_plane = fixed_coord
        for z in axis2_vals:
            for y in axis1_vals:
                coords.append([x_plane, y, z])

    else:
        raise ValueError("Plane must be 'xy', 'xz', or 'yz'.")

    # Evaluate magnetic field
    B_vals = rad.Fld(field_object, "b", coords)

    # Reshape into (n2, n1, 3)
    B_array = np.array(B_vals).reshape((n2, n1, 3))
    return B_array, axis1_vals, axis2_vals


def export_plane(B_array, axis1_vals, axis2_vals, filename):
    """
    Export vector field plane to compressed file.
    """
    np.savez_compressed(
        filename,
        B=B_array,
        axis1=axis1_vals,
        axis2=axis2_vals,
    )