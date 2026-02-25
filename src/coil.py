import radia as rad

def build_coil(x_positions, lx=20.0):
    """Builds coil as straight-wire approximation."""
    wires = []
    for idx, x in enumerate(x_positions):
        current = 1 if idx <= 2 else -1
        wires.append(rad.ObjFlmCur([[x,-lx/2,0], [x,lx/2,0]], current))
    coil = rad.ObjCnt(wires)
    rad.TrfZerPara(coil, [0,0,0], [1,0,0])
    return coil