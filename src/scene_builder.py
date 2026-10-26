"""Builds the Genesis scene: ground, Franka Panda arm, colored cubes, and colored target zones."""

import genesis as gs

from src.config import COLORS, CUBE_SIZE, CUBE_START_POS, ZONE_SIZE, ZONE_POS


def build_scene(show_viewer: bool = False, res=(960, 540)):
    scene = gs.Scene(
        show_viewer=show_viewer,
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.6, -1.6, 1.4),
            camera_lookat=(0.4, 0.0, 0.1),
            camera_fov=40,
        ),
    )

    scene.add_entity(gs.morphs.Plane())

    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

    cubes = {}
    for color, pos in CUBE_START_POS.items():
        cubes[color] = scene.add_entity(
            gs.morphs.Box(pos=pos, size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)),
            material=gs.materials.Rigid(friction=2.0),
            surface=gs.surfaces.Default(color=COLORS[color]),
        )

    zones = {}
    for color, pos in ZONE_POS.items():
        zones[color] = scene.add_entity(
            gs.morphs.Box(pos=pos, size=ZONE_SIZE, fixed=True),
            surface=gs.surfaces.Default(color=COLORS[color], opacity=0.75),
        )

    cam = scene.add_camera(
        res=res,
        pos=(1.5, -1.5, 1.6),
        lookat=(0.4, 0.0, 0.2),
        fov=45,
        GUI=False,
    )

    scene.build()

    return scene, franka, cubes, zones, cam
