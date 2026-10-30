"""Deployment: runs a trained ACT policy in closed loop inside Genesis. Claude (via Bedrock)
still parses the natural-language instruction into (object_color, target_color); from there,
the learned policy -- not the scripted IK controller -- looks at the camera image(s) and
proprioception and decides how to move the arm. The instruction text is burned into the
video so successes and failures are both clearly labeled.

Usage: run_act_demo.py --checkpoint outputs/act_policy_135_wrist.pt --wrist --video outputs/act_demo_135_wrist.mp4
"""

import argparse
import json

import cv2
import numpy as np
import torch
import genesis as gs

from src.act_model import ACTPolicy
from src.config import CUBE_START_POS
from src.demo_recorder import CHUNK_SIZE, IMG_SIZE, goal_vector
from src.llm_instruction_parser import parse_instruction
from src.pick_and_place_controller import init_robot, teleport_reset
from src.scene_builder import build_scene
from src.video_recorder import CaptionedRecorder

SIM_DT = 0.01
FPS = 30
DECISIONS_PER_EPISODE = 27  # ~= average chunk count per expert episode seen during collection
IDLE_STEPS = 60

ALL_DOFS = np.arange(9)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--wrist", action="store_true")
    parser.add_argument("--video", type=str, default=None)
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = "outputs/act_policy_wrist.pt" if args.wrist else "outputs/act_policy_top.pt"
    if args.video is None:
        args.video = "outputs/act_pick_and_place_demo_wrist.mp4" if args.wrist else "outputs/act_pick_and_place_demo_top.mp4"
    return args


def get_image(cam):
    rgb, _, _, _ = cam.render()
    return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def run_episode(policy, stats, device, franka, scene, top_cam, wrist_cam, object_color, target_color, use_wrist, recorder):
    goal = goal_vector(object_color, target_color)
    action_mean, action_std, proprio_mean, proprio_std = stats

    for _ in range(DECISIONS_PER_EPISODE):
        wrist_cam.move_to_attach()
        top_img = get_image(top_cam)
        wrist_img = get_image(wrist_cam)
        proprio = franka.get_dofs_position(dofs_idx_local=ALL_DOFS).cpu().numpy().astype(np.float32)
        state = np.concatenate([proprio, goal]).astype(np.float32)

        with torch.no_grad():
            top_t = torch.from_numpy(top_img).unsqueeze(0).to(device)
            wrist_t = torch.from_numpy(wrist_img).unsqueeze(0).to(device)
            state_t = torch.from_numpy(state).unsqueeze(0).to(device)
            state_t[:, :9] = (state_t[:, :9] - proprio_mean) / proprio_std
            images = [top_t, wrist_t] if use_wrist else [top_t]
            action_chunk_norm = policy(images, state_t)[0]  # (chunk_size, 9), normalized
            action_chunk = (action_chunk_norm * action_std + action_mean).cpu().numpy()

        for a in action_chunk:
            franka.control_dofs_position(a, dofs_idx_local=ALL_DOFS)
            scene.step()
            recorder.on_step()


def main():
    args = parse_args()
    gs.init(backend=gs.gpu, logging_level="warning")
    scene, franka, cubes, zones, cam, top_cam, wrist_cam = build_scene()
    init_robot(franka)
    scene.step()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    policy = ACTPolicy(chunk_size=CHUNK_SIZE).to(device)
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()
    stats = (
        checkpoint["action_mean"].to(device),
        checkpoint["action_std"].to(device),
        checkpoint["proprio_mean"].to(device),
        checkpoint["proprio_std"].to(device),
    )
    print("Loaded ACT policy from", args.checkpoint, "on", device, " USE_WRIST:", args.wrist)

    with open("commands.json", encoding="utf-8") as f:
        commands = json.load(f)

    recorder = CaptionedRecorder(cam, dt=SIM_DT, fps=FPS)

    recorder.set_caption("ACT policy (learned from 135 scripted demonstrations)")
    for _ in range(IDLE_STEPS):
        recorder.on_step()

    results = []
    for instruction in commands:
        print("instruction:", instruction)
        action = parse_instruction(instruction)
        print("  -> parsed action:", action)
        object_color, target_color = action["object_color"], action["target_color"]
        recorder.set_caption(
            f'Instruction: "{instruction}"  ->  {object_color} cube to {target_color} zone'
        )
        start_pos = cubes[object_color].get_pos().cpu().numpy().copy()
        zone_pos = zones[target_color].get_pos().cpu().numpy().copy()

        run_episode(
            policy, stats, device, franka, scene, top_cam, wrist_cam,
            object_color, target_color, args.wrist, recorder,
        )

        # Log this episode's outcome -- measured right here, in the exact same scene/process
        # that produced the submitted video.
        end_pos = cubes[object_color].get_pos().cpu().numpy().copy()
        moved_dist = np.linalg.norm(end_pos[:2] - start_pos[:2])
        dist_to_zone = np.linalg.norm(end_pos[:2] - zone_pos[:2])
        success = dist_to_zone < 0.06 and end_pos[2] > 0.025
        results.append((object_color, target_color, moved_dist, dist_to_zone, end_pos[2], success))
        print(
            f"  [outcome] {object_color} -> {target_color}: moved {moved_dist*100:.1f}cm, "
            f"{dist_to_zone*100:.1f}cm from zone center, final_z={end_pos[2]*100:.1f}cm SUCCESS={success}"
        )

        # Reset by directly teleporting the arm and every cube back to their canonical
        # positions -- no IK here (IK is only for generating expert demonstrations in
        # collect_demos.py). This is not recorded into the video (jump-cut to the next
        # instruction), and resetting every cube (not just the one just used) rules out any
        # incidental bump to the others during the policy's rollout carrying over unnoticed.
        teleport_reset(franka, cubes, CUBE_START_POS)

    n_success = sum(1 for r in results if r[-1])
    print(f"[summary] {n_success}/{len(results)} succeeded (this run)")

    recorder.set_caption("")
    for _ in range(IDLE_STEPS):
        recorder.on_step()

    recorder.save(args.video)
    print("Saved video to", args.video)


if __name__ == "__main__":
    main()
