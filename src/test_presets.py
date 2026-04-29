from main import CircularComponent, CoilConfig, MagnetPreset


STEEL_WALL = CircularComponent(
        inner_radius=25.5,
        outer_radius=35.4,
        lz=12.9,
        h=(-25.4 / 2) - (12.9 / 2),
        tseg=18,
        mg=[0, 0, 0],
    )
STEEL_BASEPLATE = CircularComponent(
        inner_radius=2,
        outer_radius=35.4,
        lz=10,
        h=-(25.4 + 5),
        tseg=18,
        mg=[0, 0, 0],
    )


test_presets = {
    "Test_A": MagnetPreset(
        name="Test_A",
        recess=4.6,
        inner_magnet=CircularComponent(1.6, 9.5, 11.0, -(11/2 + 4.6), 18, [0,0,1.235]),
        outer_magnet=CircularComponent(12.5, 25.4, 25.4, -25.4/2, 18, [0,0,1.235]),
        coil=CoilConfig([0.5, 1.8, 3.1, 6.2, 7.4, 8.6], 1.6, 4.0),
        wall=STEEL_WALL,
        baseplate=STEEL_BASEPLATE
    ),

    "Test_B": MagnetPreset(
        name="Test_B",
        recess=4.8,
        inner_magnet=CircularComponent(1.6, 9.5, 11.5, -(11.5/2 + 4.8), 18, [0,0,1.22]),
        outer_magnet=CircularComponent(12.5, 25.4, 25.4, -25.4/2, 18, [0,0,1.235]),
        coil=CoilConfig([0.7, 2.0, 3.5, 6.5, 7.8, 9.0], 1.8, 4.0),
        wall=STEEL_WALL,
        baseplate=STEEL_BASEPLATE,
    ),

    "Test_C": MagnetPreset(
        name="Test_C",
        recess=4.4,
        inner_magnet=CircularComponent(1.6, 9.5, 10.5, -(10.5/2 + 4.4), 18, [0,0,1.25]),
        outer_magnet=CircularComponent(12.5, 25.4, 26.0, -13.0, 18, [0,0,1.235]),
        coil=CoilConfig([0.5, 1.7, 3.0, 6.0, 7.2, 8.5], 1.5, 4.0),
        wall=STEEL_WALL,
        baseplate=STEEL_BASEPLATE,
    ),

    "Test_D": MagnetPreset(
        name="Test_D",
        recess=4.48,
        inner_magnet=CircularComponent(1.6, 9.5, 12.0, -(12/2 + 4.48), 18, [0,0,1.20]),
        outer_magnet=CircularComponent(12.5, 25.4, 25.4, -25.4/2, 18, [0,0,1.25]),
        coil=CoilConfig([1.0, 2.5, 4.0, 7.0, 8.5, 10.0], 2.0, 4.0),
        wall=STEEL_WALL,
        baseplate=STEEL_BASEPLATE,
    ),

    "Test_E": MagnetPreset(
        name="Test_E",
        recess=4.32,
        inner_magnet=CircularComponent(1.6, 9.5, 11.0, -(11/2 + 4.32), 18, [0,0,1.23]),
        outer_magnet=CircularComponent(12.5, 25.4, 24.8, -12.4, 18, [0,0,1.24]),
        coil=CoilConfig([0.8, 2.0, 3.2, 6.8, 8.0, 9.2], 1.7, 4.0),
        wall=STEEL_WALL,
        baseplate=STEEL_BASEPLATE,
    ),

    "Test_F": MagnetPreset(
        name="Test_F",
        recess=4.3,
        inner_magnet=CircularComponent(1.6, 9.5, 11.8, -(11.8/2 + 4.3), 18, [0,0,1.21]),
        outer_magnet=CircularComponent(12.5, 25.4, 25.8, -12.9, 18, [0,0,1.23]),
        coil=CoilConfig([1.2, 2.8, 4.4, 7.2, 8.8, 10.4], 2.1, 4.0),
        wall=STEEL_WALL,
        baseplate=STEEL_BASEPLATE,
    ),

    "Test_G": MagnetPreset(
        name="Test_G",
        recess=4.5,
        inner_magnet=CircularComponent(1.6, 9.5, 10.8, -(10.8/2 + 4.5), 18, [0,0,1.24]),
        outer_magnet=CircularComponent(12.5, 25.4, 25.2, -12.6, 18, [0,0,1.24]),
        coil=CoilConfig([0.6, 1.9, 3.3, 6.4, 7.7, 9.0], 1.6, 4.0),
        wall=STEEL_WALL,
        baseplate=STEEL_BASEPLATE,
    ),

    "Test_H": MagnetPreset(
        name="Test_H",
        recess=5.1,
        inner_magnet=CircularComponent(1.6, 9.5, 12.2, -(12.2/2 + 5.1), 18, [0,0,1.19]),
        outer_magnet=CircularComponent(12.5, 25.4, 26.2, -13.1, 18, [0,0,1.26]),
        coil=CoilConfig([1.5, 3.0, 4.5, 7.5, 9.0, 10.5], 2.2, 4.0),
        wall=STEEL_WALL,
        baseplate=STEEL_BASEPLATE,
    ),

    "Test_I": MagnetPreset(
        name="Test_I",
        recess=4.3,
        inner_magnet=CircularComponent(1.6, 9.5, 11.0, -(11/2 + 4.3), 18, [0,0,1.235]),
        outer_magnet=CircularComponent(12.5, 25.4, 25.4, -12.7, 18, [0,0,1.235]),
        coil=CoilConfig([0.5, 1.5, 2.8, 6.3, 7.6, 8.9], 1.5, 4.0),
        wall=STEEL_WALL,
        baseplate=STEEL_BASEPLATE,
    ),

    "Test_J": MagnetPreset(
        name="Test_J",
        recess=4.85,
        inner_magnet=CircularComponent(1.6, 9.5, 11.6, -(11.6/2 + 4.85), 18, [0,0,1.22]),
        outer_magnet=CircularComponent(12.5, 25.4, 25.6, -12.8, 18, [0,0,1.24]),
        coil=CoilConfig([1.0, 2.3, 3.7, 6.9, 8.2, 9.5], 1.9, 4.0),
        wall=STEEL_WALL,
        baseplate=STEEL_BASEPLATE,
    ),
}