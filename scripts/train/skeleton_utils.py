"""Skeleton NPZ utilities for SoccerNet-MVFoul foul classification.

Skeleton files live in a separate root from the video files:
  Video   : {data_root}/{Split}/action_{id}/clip_{idx}.mp4
  Skeleton: {sk_root}/{split_lower}/action_{id}/clip_{idx}.npz

Default skeleton root:
  data/SoccerNet/PDF1_skeleton_share/X-VARS/outputs/skeleton_yolo11

QC CSV files (pose_qc_{split}.csv) ship with the skeleton root.
Use load_qc_ok_set() to get (action_id, clip_id) pairs with status='ok'
and pass the result to load_skeleton() to skip bad skeletons.

NPZ arrays:
  keypoints      (T, P, 17, 3)  – raw pixel coords (x, y, confidence)
  keypoints_norm (T, P, 17, 3)  – normalized to [0,1]
  bboxes         (T, P, 4)      – x1,y1,x2,y2
  track_ids      (T, P)         – person IDs across frames
  person_scores  (T, P)         – YOLO detection confidence per person/frame
  frame_indices  (T,)           – which video frames were sampled
  video_width, video_height, fps
  action_id, clip_id, split, video_path  – scalar metadata
  T≤12, P≤10, J=17 COCO joints
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SK_ROOT_DEFAULT = Path(
    "data/SoccerNet/PDF1_skeleton_share/X-VARS/outputs/skeleton_yolo11"
)
T_FOUL = 3.0  # VARS spec: foul contact at ~3 s

COCO_J = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
_J = {name: i for i, name in enumerate(COCO_J)}


def skeleton_path(clip_path: str | Path, sk_root: str | Path = SK_ROOT_DEFAULT) -> Path:
    """Derive skeleton NPZ path from video clip path.

    Video:    .../mvfouls/Train/action_42/clip_0.mp4
    Skeleton: {sk_root}/train/action_42/clip_0.npz
    """
    p = Path(clip_path)
    split_lower = p.parent.parent.name.lower()   # Train→train
    action_dir  = p.parent.name                  # action_42
    clip_stem   = p.stem                         # clip_0
    return Path(sk_root) / split_lower / action_dir / f"{clip_stem}.npz"


def load_qc_ok_set(
    sk_root: str | Path = SK_ROOT_DEFAULT,
    split: str = "Train",
) -> set[tuple[str, str]]:
    """Return set of (action_id, clip_id) whose QC status is 'ok'.

    Example entry: ('action_42', 'clip_0')
    split: 'Train' | 'Valid' | 'Test'  (case-insensitive)
    """
    split_lower = split.lower()
    csv_path = Path(sk_root) / f"pose_qc_{split_lower}.csv"
    if not csv_path.exists():
        return set()  # no QC file → don't filter
    ok_set: set[tuple[str, str]] = set()
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status", "").strip() == "ok":
                ok_set.add((row["action_id"].strip(), row["clip_id"].strip()))
    return ok_set


def load_skeleton(
    clip_path: str | Path,
    sk_root: str | Path = SK_ROOT_DEFAULT,
    ok_set: set[tuple[str, str]] | None = None,
) -> dict | None:
    """Load skeleton NPZ for a video clip.

    Returns None if:
    - NPZ file does not exist
    - ok_set provided and (action_id, clip_id) not in it (QC failed)
    """
    spath = skeleton_path(clip_path, sk_root)
    if not spath.exists():
        return None

    # QC filter
    if ok_set is not None:
        p = Path(clip_path)
        action_id = p.parent.name          # action_42
        clip_id   = p.stem                 # clip_0
        if (action_id, clip_id) not in ok_set:
            return None

    data = np.load(str(spath), allow_pickle=False)
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Contact frame
# ---------------------------------------------------------------------------

def contact_frame_idx(sk: dict, t_foul: float = T_FOUL) -> int:
    fps    = float(sk.get("fps", 25.0))
    target = int(round(t_foul * fps))
    fi     = np.asarray(sk["frame_indices"])
    return int(np.argmin(np.abs(fi - target)))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2  = a - b, c - b
    cos_val = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(math.degrees(math.acos(np.clip(cos_val, -1.0, 1.0))))


def _describe_person(kp_norm: np.ndarray, label: str) -> str:
    """Text description of one person from (17, 3) normalized keypoints."""

    def pt(j): return kp_norm[_J[j], :2]
    def cf(j): return float(kp_norm[_J[j], 2])
    ok = lambda j: cf(j) > 0.3

    parts = [f"  {label}:"]

    for side, ankle_j, knee_j, hip_j in [
        ("left",  "left_ankle",  "left_knee",  "left_hip"),
        ("right", "right_ankle", "right_knee", "right_hip"),
    ]:
        if not ok(ankle_j):
            continue
        ankle_y = pt(ankle_j)[1]
        hip_y   = pt(hip_j)[1]
        # image coords: y=0=top, y=1=bottom; raised foot → ankle_y < hip_y
        raise_frac = max(0.0, (hip_y - ankle_y) / (abs(hip_y - ankle_y) + 1e-8)) if ok(hip_j) else 0.0
        if raise_frac > 0.3:
            height_desc = "high (knee-level or above)"
        elif raise_frac > 0.05:
            height_desc = "mid (shin-to-knee)"
        else:
            height_desc = "low (ankle-level)"

        angle_str = ""
        if ok(knee_j) and ok(hip_j):
            angle = _angle_deg(pt(hip_j), pt(knee_j), pt(ankle_j))
            angle_str = f", knee {angle:.0f}°"
        parts.append(f"    {side} foot: {height_desc}{angle_str}")

    if all(ok(j) for j in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
        mid_sh = (pt("left_shoulder") + pt("right_shoulder")) / 2
        mid_hp = (pt("left_hip")      + pt("right_hip"))      / 2
        torso  = mid_sh - mid_hp
        lean   = math.degrees(math.atan2(abs(torso[0]), abs(torso[1]) + 1e-8))
        parts.append(f"    torso lean: {lean:.0f}° from vertical")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API: text hint for VLM prompt injection (Option 2)
# ---------------------------------------------------------------------------

def skeleton_to_text(sk: dict, max_persons: int = 2) -> str:
    """Convert skeleton NPZ to a human-readable text block for VLM prompts."""
    kp_norm = np.asarray(sk["keypoints_norm"])     # (T, P, 17, 3)
    fi      = np.asarray(sk["frame_indices"])
    fps     = float(sk.get("fps", 25.0))
    ci      = contact_frame_idx(sk)
    t_contact = float(fi[ci]) / fps

    # Select top-N persons by YOLO score at contact frame (fall back to kp conf)
    if "person_scores" in sk:
        scores = np.asarray(sk["person_scores"])[ci]    # (P,)
    else:
        scores = kp_norm[ci, :, :, 2].mean(axis=1)     # (P,)

    top = [int(i) for i in np.argsort(scores)[::-1] if scores[i] > 0.1][:max_persons]
    if not top:
        return ""

    labels = ["Defender", "Attacker"] if len(top) >= 2 else ["Player"]
    lines  = [f"[Skeleton at contact moment t≈{t_contact:.1f}s (frame {fi[ci]})]"]
    for slot, p_idx in enumerate(top):
        label = labels[min(slot, len(labels) - 1)]
        lines.append(_describe_person(kp_norm[ci, p_idx], label))
    lines.append("[End skeleton]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API: feature extraction for skeleton classifier (Option 1)
# ---------------------------------------------------------------------------

def extract_features(
    sk: dict,
    n_frames: int = 12,
    max_persons: int = 2,
) -> np.ndarray:
    """Return (n_frames, max_persons * 17 * 3) float32 feature array.

    Person selection: highest mean YOLO person_scores across all frames.
    Zero-pads missing frames or persons.
    """
    kp  = np.asarray(sk["keypoints_norm"])   # (T, P, 17, 3)
    T, P = kp.shape[:2]

    if "person_scores" in sk:
        score_mean = np.asarray(sk["person_scores"]).mean(axis=0)  # (P,)
    else:
        score_mean = kp[:, :, :, 2].mean(axis=(0, 2))             # (P,)

    top_persons = [
        int(i) for i in np.argsort(score_mean)[::-1]
        if score_mean[i] > 0.05
    ][:max_persons]

    feat_dim = max_persons * 17 * 3
    out = np.zeros((n_frames, feat_dim), dtype=np.float32)
    for slot, p_idx in enumerate(top_persons):
        start = slot * 17 * 3
        for t in range(min(T, n_frames)):
            out[t, start:start + 17 * 3] = kp[t, p_idx].reshape(-1)

    return out   # (n_frames, feat_dim)


# ---------------------------------------------------------------------------
# Utility: build matched sample list with skeleton availability
# ---------------------------------------------------------------------------

def filter_samples_with_skeleton(
    samples: list[dict],
    sk_root: str | Path = SK_ROOT_DEFAULT,
    ok_only: bool = True,
) -> tuple[list[dict], dict]:
    """Return (matched_samples, stats) where matched_samples have a valid skeleton.

    Stats keys: total, matched, qc_filtered, missing_npz
    """
    sk_root = Path(sk_root)

    # Build per-split QC sets once
    qc_cache: dict[str, set] = {}

    matched, qc_filtered, missing_npz = [], 0, 0
    for s in samples:
        clip_path = Path(s["clip_path"])
        split_cap = clip_path.parent.parent.name          # 'Train' / 'Valid' / 'Test'

        if ok_only:
            if split_cap not in qc_cache:
                qc_cache[split_cap] = load_qc_ok_set(sk_root, split_cap)
            ok_set = qc_cache[split_cap]
        else:
            ok_set = None

        sk = load_skeleton(clip_path, sk_root, ok_set)
        if sk is None:
            spath = skeleton_path(clip_path, sk_root)
            if spath.exists():
                qc_filtered += 1
            else:
                missing_npz += 1
        else:
            matched.append({**s, "_skeleton": sk})

    stats = {
        "total":       len(samples),
        "matched":     len(matched),
        "qc_filtered": qc_filtered,
        "missing_npz": missing_npz,
    }
    return matched, stats
