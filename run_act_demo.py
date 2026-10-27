"""Deployment: runs a trained ACT policy in closed loop inside Genesis. Claude (via Bedrock)
still parses the natural-language instruction into (object_color, target_color); from there,
the learned policy -- not the scripted IK controller -- looks at the camera image(s) and
proprioception and decides how to move the arm.

Usage: run_act_demo.py --checkpoint outputs/act_policy_135_wrist.pt --wrist --video outputs/act_demo_135_wrist.mp4
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
import genesis as gs

from src.act_model import ACTPolicy
from src.demo_recorder import CHUNK_SIZE, IMG_SIZE, goal_vector
from src.llm_instruction_parser import parse_instruction
from src.pick_and_place_controller import go_ready, init_robot
from src.scene_builder import build_scene

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


def run_episode(policy, stats, device, franka, scene, top_cam, wrist_cam, object_color, target_color, use_wrist):
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

    os.makedirs("outputs", exist_ok=True)
    cam.start_recording(save_to_filename=args.video, fps=FPS)

    for _ in range(IDLE_STEPS):
        scene.step()

    for instruction in commands:
        print("instruction:", instruction)
        action = parse_instruction(instruction)
        print("  -> parsed action:", action)
        run_episode(
            policy, stats, device, franka, scene, top_cam, wrist_cam,
            action["object_color"], action["target_color"], args.wrist,
        )
        # Every training episode ends with (and the next one begins right after) a scripted
        # go_ready() -- without repeating that here, the next episode would start from
        # whatever odd pose the policy's own (imperfect) rollout happened to stop at, which
        # the policy never saw during training.
        go_ready(franka, scene)

    for _ in range(IDLE_STEPS):
        scene.step()

    cam.stop_recording()
    print("Saved video to", args.video)


if __name__ == "__main__":
    main()
