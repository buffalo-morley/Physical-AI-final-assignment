"""Builds the Genesis scene: ground, Franka Panda arm, colored cubes, and colored target zones."""

import numpy as np
import genesis as gs

from src.config import COLORS, CUBE_SIZE, CUBE_START_POS, ZONE_SIZE, ZONE_POS

def _look_at_transform(eye, target, up_hint=(0.0, 1.0, 0.0)):
    """Builds a 4x4 transform (OpenGL convention: camera looks along local -Z, local +Y is
    up) placed at `eye` and aimed at `target`, both given in the parent link's local frame."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up_hint = np.asarray(up_hint, dtype=np.float64)
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = true_up
    T[:3, 2] = -forward
    T[:3, 3] = eye
    return T


# Wrist camera mount, expressed as a 4x4 transform relative to the "hand" link.
# In the hand's local frame, +Z points along the grasp/approach direction (where the fingers
# point) and Y is the finger-opening axis -- so neither is a valid mounting surface (Z points
# straight down the gap between the fingers and whatever they're grasping; Y runs into the
# fingers themselves). A real eye-in-hand camera (e.g. a RealSense bolted to a Panda wrist)
# is instead mounted on the flat side of the hand housing along local X, angled forward and
# inward to look past the fingers at the grasp point ahead of them.
WRIST_CAM_OFFSET_T = _look_at_transform(eye=(0.13, 0.0, 0.0), target=(0.0, 0.0, 0.11))


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

    # Cinematic camera, used only for the human-facing demo video.
    cam = scene.add_camera(
        res=res,
        pos=(1.5, -1.5, 1.6),
        lookat=(0.4, 0.0, 0.2),
        fov=45,
        GUI=False,
    )

    # Static overhead ("ceiling") camera -- the standard fixed third-person view for
    # tabletop manipulation: looks straight down over the workspace, so the arm doesn't
    # occlude the objects the way an oblique cinematic angle does.
    top_cam = scene.add_camera(
        res=(128, 128),
        pos=(0.45, 0.0, 0.8),
        lookat=(0.45, 0.0, 0.0),
        up=(1.0, 0.0, 0.0),
        fov=38,
        GUI=False,
    )

    # Wrist ("eye-in-hand") camera, attached to the gripper -- see comment on
    # WRIST_CAM_OFFSET_T above.
    wrist_cam = scene.add_camera(
        res=(128, 128),
        pos=(0.35, -0.2, 0.3),  # placeholder pose; overwritten by move_to_attach() below
        lookat=(0.35, -0.2, 0.0),
        fov=70,
        near=0.01,  # the gripper gets within a few cm of the ground/object during a grasp,
        # closer than the default 0.1m near-clip plane -- without this the wrist view goes
        # completely black right when it matters most.
        GUI=False,
    )

    scene.build()

    wrist_cam.attach(franka.get_link("hand"), WRIST_CAM_OFFSET_T)
    wrist_cam.move_to_attach()

    return scene, franka, cubes, zones, cam, top_cam, wrist_cam
