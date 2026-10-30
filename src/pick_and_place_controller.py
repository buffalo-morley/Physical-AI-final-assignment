"""Low-level Franka Panda motion primitives: smooth IK-based moves, gripper control, and a
physically-simulated pick-and-place routine (grasp via finger friction, not teleportation).

Every motion primitive accepts an optional `on_step(action_vector)` callback, invoked once per
simulation step with the 9-dim joint target that was just commanded. This is used by
`collect_demos.py` to record (image, proprioception, action) tuples from this scripted
"expert" controller, which are then used to train the ACT imitation-learning policy.
"""

import numpy as np

from src.config import (
    APPROACH_STEPS,
    CUBE_SIZE,
    GRASP_STEPS,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    MOVE_STEPS,
    SETTLE_STEPS,
    ZONE_SIZE,
)

CUBE_HALF = CUBE_SIZE / 2
ZONE_HALF = ZONE_SIZE[2] / 2

ALL_DOFS = np.arange(9)
ARM_DOFS = np.arange(7)
FINGER_DOFS = np.array([7, 8])

KP = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 1200, 1200], dtype=np.float64)
KV = np.array([450, 450, 350, 350, 200, 200, 200, 40, 40], dtype=np.float64)

READY_QPOS = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.2, 0.78, 0.04, 0.04])

DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # gripper fingers pointing straight down

# Vertical offset from the "hand" link frame to the mid-height grasp point between the
# fingertips, calibrated against the 4cm cubes used in this scene.
GRASP_Z_OFFSET = 0.103


def init_robot(franka):
    """Set PD gains and bring the arm to a neutral ready pose.

    `set_dofs_position` only teleports the kinematic state; it does not touch the PD
    controller's target. Without also setting that target to the same pose, the very next
    `scene.step()` would have the (huge) arm gains pulling against whatever stale/default
    target was there before, snapping the arm violently before any real motion command is
    ever issued.
    """
    franka.set_dofs_kp(KP, dofs_idx_local=ALL_DOFS)
    franka.set_dofs_kv(KV, dofs_idx_local=ALL_DOFS)
    franka.set_dofs_position(READY_QPOS, dofs_idx_local=ALL_DOFS)
    franka.control_dofs_position(READY_QPOS, dofs_idx_local=ALL_DOFS)


def teleport_reset(franka, cubes, cube_start_pos):
    """Directly teleports the arm to the ready pose and every cube back to its canonical
    start position -- no IK, no simulated motion. IK (via pick()/place()) is only meant to
    generate expert demonstrations in collect_demos.py; evaluating or deploying the ACT
    policy should never rely on IK to clean up between trials, since that would itself be
    another fallible physical action and could mask or mimic the policy's own errors. Also
    zeroes every DOF's velocity so no residual motion carries into the next trial."""
    init_robot(franka)
    for color, pos in cube_start_pos.items():
        cubes[color].set_pos(pos)
        cubes[color].set_quat([1.0, 0.0, 0.0, 0.0])
        cubes[color].zero_all_dofs_velocity()


def _current_arm_qpos(franka):
    return franka.get_dofs_position(dofs_idx_local=ARM_DOFS).cpu().numpy()


def _current_gripper_width(franka):
    return float(franka.get_dofs_position(dofs_idx_local=FINGER_DOFS).cpu().numpy().mean())


def _interp_run(franka, scene, arm_target, gripper_target, steps, on_step=None):
    arm_start = _current_arm_qpos(franka)
    grip_start = _current_gripper_width(franka)
    for i in range(1, steps + 1):
        alpha = i / steps
        arm_q = arm_start + (arm_target - arm_start) * alpha
        grip_q = grip_start + (gripper_target - grip_start) * alpha
        full = np.concatenate([arm_q, [grip_q, grip_q]])
        franka.control_dofs_position(full, dofs_idx_local=ALL_DOFS)
        scene.step()
        if on_step is not None:
            on_step(full)


def _hold(franka, scene, arm_target, gripper_target, steps, on_step=None):
    full = np.concatenate([arm_target, [gripper_target, gripper_target]])
    for _ in range(steps):
        franka.control_dofs_position(full, dofs_idx_local=ALL_DOFS)
        scene.step()
        if on_step is not None:
            on_step(full)


def move_to(
    franka,
    scene,
    pos,
    quat=DOWN_QUAT,
    gripper=None,
    move_steps=MOVE_STEPS,
    settle_steps=SETTLE_STEPS,
    cartesian_waypoints=8,
    on_step=None,
):
    """Move the end-effector to `pos`/`quat`.

    The straight-line Cartesian path is split into several waypoints, each solved with its
    own IK call (seeded from the previous solution). This keeps the gripper's orientation
    level throughout the transit instead of just interpolating the start/end joint angles,
    which can otherwise swing the wrist through a very different configuration mid-flight
    and shake a grasped object loose.
    """
    hand = franka.get_link("hand")
    if gripper is None:
        gripper = _current_gripper_width(franka)

    start_pos = hand.get_pos().cpu().numpy()
    target_pos = np.asarray(pos, dtype=np.float64)
    grip_start = _current_gripper_width(franka)
    steps_per_wp = max(1, move_steps // cartesian_waypoints)

    arm_target = _current_arm_qpos(franka)
    for wp in range(1, cartesian_waypoints + 1):
        alpha = wp / cartesian_waypoints
        wp_pos = start_pos + (target_pos - start_pos) * alpha
        wp_grip = grip_start + (gripper - grip_start) * alpha
        init_qpos = np.concatenate([arm_target, [wp_grip, wp_grip]])
        qpos_full = franka.inverse_kinematics(
            link=hand, pos=wp_pos, quat=quat, dofs_idx_local=ARM_DOFS, init_qpos=init_qpos
        )
        arm_target = qpos_full.cpu().numpy()[:7]
        _interp_run(franka, scene, arm_target, wp_grip, steps_per_wp, on_step=on_step)

    if settle_steps:
        _hold(franka, scene, arm_target, gripper, settle_steps, on_step=on_step)


def set_gripper(franka, scene, width, steps=GRASP_STEPS, on_step=None):
    """Open/close the gripper in place without moving the arm."""
    arm_target = _current_arm_qpos(franka)
    _interp_run(franka, scene, arm_target, width, steps, on_step=on_step)
    _hold(franka, scene, arm_target, width, 10, on_step=on_step)


def pick(franka, scene, cube_pos, approach_height=0.18, on_step=None):
    """Grasp the object located at `cube_pos` (physics-based grip via finger friction)."""
    x, y, z = float(cube_pos[0]), float(cube_pos[1]), float(cube_pos[2])
    grasp_pos = (x, y, z + GRASP_Z_OFFSET)
    approach_pos = (x, y, z + GRASP_Z_OFFSET + approach_height)

    move_to(franka, scene, approach_pos, gripper=GRIPPER_OPEN, on_step=on_step)
    move_to(franka, scene, grasp_pos, gripper=GRIPPER_OPEN, move_steps=APPROACH_STEPS, on_step=on_step)
    set_gripper(franka, scene, GRIPPER_CLOSE, on_step=on_step)
    move_to(franka, scene, approach_pos, gripper=GRIPPER_CLOSE, move_steps=APPROACH_STEPS, on_step=on_step)


def place(franka, scene, target_pos, approach_height=0.18, surface_half_height=ZONE_HALF, on_step=None):
    """Lower the currently-held object onto a surface at `target_pos` and release it.

    `surface_half_height` is the half-thickness of whatever the object will rest on:
    use ZONE_HALF (default) for a colored zone pad, or 0.0 to place directly on the ground.
    """
    x, y, z = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
    final_cube_z = z + surface_half_height + CUBE_HALF
    drop_pos = (x, y, final_cube_z + GRASP_Z_OFFSET)
    approach_pos = (x, y, final_cube_z + GRASP_Z_OFFSET + approach_height)

    move_to(franka, scene, approach_pos, gripper=GRIPPER_CLOSE, on_step=on_step)
    move_to(franka, scene, drop_pos, gripper=GRIPPER_CLOSE, move_steps=APPROACH_STEPS, on_step=on_step)
    set_gripper(franka, scene, GRIPPER_OPEN, on_step=on_step)
    move_to(franka, scene, approach_pos, gripper=GRIPPER_OPEN, move_steps=APPROACH_STEPS, on_step=on_step)


def go_ready(franka, scene, steps=MOVE_STEPS, on_step=None):
    _interp_run(franka, scene, READY_QPOS[:7], GRIPPER_OPEN, steps, on_step=on_step)
