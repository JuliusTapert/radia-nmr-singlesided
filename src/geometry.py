import math
import radia as rad

def build_ring_geometry(inner_radius, outer_radius, lz, h, tseg, mg=[0,0,0]):
    #Build a segmented ring (magnet or steel) from an arbitrarily angular trapezoid based on segmentation

    theta_segment = math.pi / (2 * tseg)
    ring_wedges = []
    for i in range(tseg):
        phi0 = i * theta_segment
        phi1 = (i+1) * theta_segment

        #precess vertices around unit circle (outer then inner in reverse)
        verts = [
            [outer_radius * math.cos(phi0), outer_radius * math.sin(phi0)],
            [outer_radius * math.cos(phi1), outer_radius * math.sin(phi1)],
            [inner_radius * math.cos(phi1), inner_radius * math.sin(phi1)],
            [inner_radius * math.cos(phi0), inner_radius * math.sin(phi0)]
        ]
        wedge = rad.ObjThckPgn(h, lz, verts, 'z', mg)
        color = [0.5,0.5,0.5] if mg==[0,0,0] else [0.9,0.2,0.1]
        rad.ObjDrwAtr(wedge, color)
        ring_wedges.append(wedge)
    return rad.ObjCnt(ring_wedges)


def build_steel_assembly(steel_baseplate, steel_wall, steelmat):
    """Builds steel assembly as a single Radia object and applies material."""
    baseplate_obj = build_ring_geometry(**steel_baseplate.as_dict())
    wall_obj = build_ring_geometry(**steel_wall.as_dict())
    steel_assembly = rad.ObjCnt([baseplate_obj, wall_obj])
    rad.MatApl(steel_assembly, steelmat)
    return steel_assembly