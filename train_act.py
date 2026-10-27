"""Trains the ACT-style imitation-learning policy on a dataset collected by collect_demos.py
(scripted-expert demonstrations: images + proprioception+goal -> action chunk).

Actions (and the proprioception part of the state) are z-score normalized before computing
the loss. Without this, the MSE loss is dominated by the arm joints (range roughly -3..3 rad)
and almost completely ignores the finger dofs (range 0..0.04) -- which is exactly the part of
the action that determines whether the gripper actually grasps the object. The normalization
stats are saved alongside the checkpoint and must be used to de-normalize the policy's output
at inference time (see run_act_demo.py).

Usage: train_act.py --dataset outputs/act_dataset_135.pt --checkpoint outputs/act_policy_135_wrist.pt --wrist
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.act_model import ACTPolicy
from src.demo_recorder import CHUNK_SIZE

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
PROPRIO_DIM = 9  # leading slice of `state` that gets normalized (goal one-hot is left as-is)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="outputs/act_dataset.pt")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--wrist", action="store_true")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = "outputs/act_policy_wrist.pt" if args.wrist else "outputs/act_policy_top.pt"
    return args


class DemoDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        top_image = torch.from_numpy(s["top_image"])
        wrist_image = torch.from_numpy(s["wrist_image"])
        state = torch.from_numpy(s["state"])
        actions = torch.from_numpy(s["actions"])
        return top_image, wrist_image, state, actions


def compute_stats(samples):
    all_actions = np.concatenate([s["actions"] for s in samples], axis=0)  # (N, 9)
    all_proprio = np.stack([s["state"][:PROPRIO_DIM] for s in samples], axis=0)  # (N, 9)
    action_mean, action_std = all_actions.mean(0), all_actions.std(0) + 1e-6
    proprio_mean, proprio_std = all_proprio.mean(0), all_proprio.std(0) + 1e-6
    return (
        torch.from_numpy(action_mean.astype(np.float32)),
        torch.from_numpy(action_std.astype(np.float32)),
        torch.from_numpy(proprio_mean.astype(np.float32)),
        torch.from_numpy(proprio_std.astype(np.float32)),
    )


def main():
    args = parse_args()
    use_wrist = args.wrist
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, " dataset:", args.dataset, " USE_WRIST:", use_wrist, " checkpoint:", args.checkpoint)

    samples = torch.load(args.dataset, weights_only=False)
    print(f"loaded {len(samples)} samples")

    action_mean, action_std, proprio_mean, proprio_std = compute_stats(samples)
    action_mean, action_std = action_mean.to(device), action_std.to(device)
    proprio_mean, proprio_std = proprio_mean.to(device), proprio_std.to(device)

    dataset = DemoDataset(samples)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    policy = ACTPolicy(chunk_size=CHUNK_SIZE).to(device)
    optim = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-4)

    for epoch in range(1, EPOCHS + 1):
        losses = []
        for top_image, wrist_image, state, actions in loader:
            top_image = top_image.to(device)
            wrist_image = wrist_image.to(device)
            state = state.to(device)
            actions = actions.to(device)

            state = state.clone()
            state[:, :PROPRIO_DIM] = (state[:, :PROPRIO_DIM] - proprio_mean) / proprio_std
            actions_norm = (actions - action_mean) / action_std

            images = [top_image, wrist_image] if use_wrist else [top_image]
            pred = policy(images, state)
            loss = torch.nn.functional.mse_loss(pred, actions_norm)

            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

        print(f"epoch {epoch}/{EPOCHS}  loss={np.mean(losses):.6f}")

    torch.save(
        {
            "model_state": policy.state_dict(),
            "use_wrist": use_wrist,
            "action_mean": action_mean.cpu(),
            "action_std": action_std.cpu(),
            "proprio_mean": proprio_mean.cpu(),
            "proprio_std": proprio_std.cpu(),
        },
        args.checkpoint,
    )
    print(f"Saved policy checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
