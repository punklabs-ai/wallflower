"""viz.analyze_motion — offline analysis of a live_motion JSONL session log.

Reports, per ground-truth label (still/moving/unlabeled):
  * motion-energy distribution (percentiles),
  * presence rate  (== false-positive rate when label is 'still';
                     == detection rate    when label is 'moving'),
and suggests a `k` (threshold = floor + k*MAD) that would hold the resting
false-positive rate at/below a target, plus the separation between still & moving.
Optionally renders a PNG: motion + threshold + presence, shaded by label.

Usage:
    .venv/bin/python -m viz.analyze_motion [SESSION.jsonl] [--target-fp 0.01] [--plot out.png]
(no path -> newest session in data/reports/motion_logs/)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def load(path: Path):
    meta, rows = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("type") == "meta":
            meta = d
        elif d.get("type") == "s":
            rows.append(d)
    return meta, rows


def pct(a, ps):
    return {p: round(float(np.percentile(a, p)), 4) for p in ps} if len(a) else {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m viz.analyze_motion")
    ap.add_argument("session", nargs="?", default=None)
    ap.add_argument("--logdir", default="data/reports/motion_logs")
    ap.add_argument("--target-fp", type=float, default=0.01,
                    help="desired resting false-positive rate to size k")
    ap.add_argument("--plot", default=None, help="write a PNG here")
    args = ap.parse_args(argv)

    path = Path(args.session) if args.session else None
    if path is None:
        cands = sorted(glob.glob(os.path.join(args.logdir, "session_*.jsonl")))
        if not cands:
            print(f"no session logs in {args.logdir}")
            return 1
        path = Path(cands[-1])
    meta, rows = load(path)
    if not rows:
        print(f"{path}: no samples")
        return 1

    t = np.array([r["t"] for r in rows], float)
    motion = np.array([r["motion"] for r in rows], float)
    thr = np.array([r["thr"] for r in rows], float)
    floor = np.array([r["floor"] for r in rows], float)
    presence = np.array([r["presence"] for r in rows], bool)
    labels = np.array([r.get("label", "unlabeled") for r in rows])
    dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    print(f"=== {path.name} ===")
    print(f"samples={len(rows)}  duration={dur:.1f}s  "
          f"mean_fps={np.mean([r['fps'] for r in rows]):.0f}  cfg_k={meta.get('k')}")
    print(f"{'label':<11}{'sec':>7}{'presence%':>11}{'motion p50':>12}"
          f"{'p95':>9}{'p99':>9}{'max':>9}")
    seg = {}
    for lab in ("still", "moving", "unlabeled"):
        mask = labels == lab
        if not mask.any():
            continue
        m = motion[mask]
        secs = mask.sum() / float(meta.get("fs", 20.0))
        presrate = 100.0 * presence[mask].mean()
        p = pct(m, [50, 95, 99])
        seg[lab] = m
        print(f"{lab:<11}{secs:>7.1f}{presrate:>10.1f}%{p.get(50,0):>12.4f}"
              f"{p.get(95,0):>9.4f}{p.get(99,0):>9.4f}{float(m.max()):>9.4f}")

    # Suggest k from the 'still' distribution: choose threshold at the (1-target_fp)
    # quantile of resting motion, then express it relative to median+MAD.
    if "still" in seg and len(seg["still"]) > 20:
        s = seg["still"]
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med))) * 1.4826 or 1e-3
        q = float(np.quantile(s, 1.0 - args.target_fp))
        k_sugg = max(0.0, (q - med) / mad)
        print(f"\nresting: median={med:.4f} MAD={mad:.4f}  "
              f"{(1-args.target_fp)*100:.0f}th-pct={q:.4f}")
        print(f"SUGGESTED --k {k_sugg:.1f}  (for ~{args.target_fp*100:.1f}% "
              f"false-positive at rest)")
        if "moving" in seg and len(seg["moving"]) > 5:
            mv = seg["moving"]
            sep = float(np.median(mv)) / (med + 1e-9)
            detect = 100.0 * (mv > (med + k_sugg * mad)).mean()
            print(f"moving/still median ratio={sep:.1f}x  "
                  f"-> at suggested k, moving detected ~{detect:.0f}% of the time")
    else:
        print("\n(label some 'still' time in the UI to auto-size k)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tt = t - t[0]
        fig, ax = plt.subplots(figsize=(12, 4))
        colmap = {"still": "#1f9e4d33", "moving": "#f38ba833"}
        # shade labeled spans
        cur = labels[0]; start = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != cur:
                if cur in colmap:
                    ax.axvspan(tt[start], tt[i - 1], color=colmap[cur], lw=0)
                if i < len(labels):
                    cur = labels[i]; start = i
        ax.plot(tt, motion, color="#89b4fa", lw=1.0, label="motion energy")
        ax.plot(tt, thr, color="#f38ba8", lw=1.0, ls="--", label="threshold")
        ax.plot(tt, floor, color="#7f849c", lw=0.8, ls=":", label="noise floor")
        ax.fill_between(tt, 0, motion.max() * 1.05, where=presence,
                        color="#f9e2af22", lw=0, label="presence")
        ax.set_xlabel("time (s)"); ax.set_ylabel("motion energy")
        ax.set_title(f"{path.name}  (green=still, red=moving)")
        ax.legend(loc="upper right", fontsize=8); ax.set_ylim(0, None)
        fig.tight_layout(); fig.savefig(args.plot, dpi=110)
        print(f"plot -> {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
