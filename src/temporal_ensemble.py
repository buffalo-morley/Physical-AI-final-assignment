"""Temporal ensembling for ACT-style chunked action prediction (Zhao et al., 2023).

Without this, executing a predicted chunk of `chunk_size` actions in full before replanning
causes a visible discontinuity ("jolt") every time control jumps from the tail of one chunk
to the head of a freshly-predicted one -- the two chunks were predicted independently and
have no reason to agree at that boundary. Temporal ensembling instead replans every
`replan_every` steps and blends the overlapping predictions for the step actually being
executed, weighted so that the freshest prediction dominates but older ones still smooth the
transition.
"""

import numpy as np

REPLAN_EVERY = 10  # steps (0.1s at dt=0.01) between re-predicting a chunk


class TemporalEnsembler:
    def __init__(self, m=0.01):
        self.m = m
        self.buffer = []  # list of (chunk: np.ndarray (chunk_size, action_dim), start_t: int)

    def update(self, chunk, current_t):
        self.buffer.append((chunk, current_t))
        chunk_size = chunk.shape[0]
        self.buffer = [(c, s) for c, s in self.buffer if s + chunk_size > current_t]

    def get_action(self, t):
        preds, ages = [], []
        for chunk, start in self.buffer:
            idx = t - start
            if 0 <= idx < chunk.shape[0]:
                preds.append(chunk[idx])
                ages.append(t - start)  # 0 = predicted this step, larger = staler prediction

        weights = np.exp(-self.m * np.array(ages))
        weights = weights / weights.sum()
        preds = np.stack(preds)
        return (weights[:, None] * preds).sum(axis=0)
