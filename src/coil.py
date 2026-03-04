import radia as rad

def build_coil(x_positions, I=1.0, lx=20.0):
    """Builds coil as straight-wire approximation."""
    wires = []

    for idx, x in enumerate(x_positions):
        current = (1*I) if idx <= 2 else (-1*I)
        wires.append(rad.ObjFlmCur([[x,-lx/2,0], [x,lx/2,0]], current))
        
    coil = rad.ObjCnt(wires)
    rad.TrfZerPara(coil, [0,0,0], [1,0,0])
    return coil