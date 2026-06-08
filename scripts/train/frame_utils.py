"""Foul-anchored frame extraction for training and evaluation.

VARS spec: foul occurs at ~3.0 s into the 5-second clip (60% mark), NOT at
the center. This module provides a drop-in replacement for the standard
{"type": "video", "video": path, "nframes": N} content entry.

Usage in content builders:
    from train.frame_utils import make_video_entry
    content.append(make_video_entry(vid_path, num_frames))
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Timestamp strategy
# ---------------------------------------------------------------------------

T_FOUL  = 3.0   # VARS spec: foul at ~3 s in a 5-s clip
T_START = 0.5   # skip first ~0.5 s (players still far away)
T_END   = 4.5   # skip last ~0.5 s (aftermath already settled)


def foul_frame_index(
    n: int,
    t_start: float = T_START,
    t_end: float   = T_END,
    t_foul: float  = T_FOUL,
) -> int:
    """Return the 1-based index of the foul frame among n foul-anchored frames.

    Example: foul_frame_index(8)  → 5   (frame  5/8  ≈ t=3.0s)
             foul_frame_index(16) → 10  (frame 10/16 ≈ t=3.0s)
             foul_frame_index(32) → 20  (frame 20/32 ≈ t=3.0s)
    """
    span     = t_end - t_start
    ratio    = (t_foul - t_start) / span if span > 0 else 0.5
    n_before = max(1, round(ratio * (n - 1)))
    return n_before + 1  # convert to 1-based


def foul_anchored_timestamps(
    n: int,
    t_start: float = T_START,
    t_end: float   = T_END,
    t_foul: float  = T_FOUL,
) -> list[float]:
    """Return n timestamps that always include t_foul.

    The remaining (n-1) frames are distributed proportionally before and
    after the foul, using the fraction (t_foul-t_start)/(t_end-t_start).

    Example — n=16, [0.5, 4.5], foul=3.0:
      ratio = 2.5/4.0 = 0.625
      n_before=9, n_after=6
      → [0.50, 0.78, …, 2.72, 3.00★, 3.25, 3.50, …, 4.50]
    """
    t_foul = max(t_start, min(t_end, t_foul))
    span   = t_end - t_start
    ratio  = (t_foul - t_start) / span if span > 0 else 0.5

    n_before = max(1, round(ratio * (n - 1)))   # frames from t_start…t_foul (inclusive)
    n_after  = n - 1 - n_before                 # frames after t_foul

    before = [t_start + (t_foul - t_start) * i / n_before
              for i in range(n_before + 1)]
    after  = [t_foul + (t_end - t_foul) * i / max(n_after, 1)
              for i in range(1, n_after + 1)]
    return before + after


# ---------------------------------------------------------------------------
# Frame loader (torchcodec → PIL)
# ---------------------------------------------------------------------------

def _load_pil_frames(video_path: str, timestamps: list[float]):
    """Extract PIL frames at given timestamps via torchcodec.

    Returns list[PIL.Image] on success, None on any failure.
    Callers should fall back to path-based loading on None.
    """
    try:
        from torchcodec.decoders import VideoDecoder
        from PIL import Image as PILImage

        decoder = VideoDecoder(video_path)
        meta    = decoder.metadata

        # Resolve fps and total frames
        fps   = float(getattr(meta, "average_fps", None) or
                      getattr(meta, "best_video_stream_fps", None) or 25.0)
        total = int(getattr(meta, "num_frames", None) or
                    round(float(getattr(meta, "duration_seconds", None) or 5.0) * fps))
        total = max(1, total)

        indices = [max(0, min(total - 1, round(t * fps))) for t in timestamps]
        result  = decoder.get_frames_at(indices=indices)
        frames  = result.data   # (N, C, H, W) uint8 tensor, values 0-255

        pils = []
        for i in range(frames.shape[0]):
            arr = frames[i].permute(1, 2, 0).cpu().numpy()
            pils.append(PILImage.fromarray(arr))
        return pils

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_video_entry(
    video_path: str | Path,
    num_frames: int,
    t_start: float = T_START,
    t_end:   float = T_END,
    t_foul:  float = T_FOUL,
) -> dict[str, Any]:
    """Return a content-dict entry for the Qwen3VL processor.

    Tries to extract frames at foul-anchored timestamps via torchcodec.
    Falls back to {"video": path, "nframes": N} (uniform sampling) on failure.

    The returned dict is a drop-in replacement for:
        {"type": "video", "video": str(path), "nframes": num_frames}
    """
    timestamps = foul_anchored_timestamps(num_frames, t_start, t_end, t_foul)
    pils       = _load_pil_frames(str(video_path), timestamps)

    if pils is not None and len(pils) == num_frames:
        # PIL list: processor uses all frames as-is (no further sampling)
        return {"type": "video", "video": pils}
    else:
        # Fallback: processor will sample uniformly (old behaviour)
        return {"type": "video", "video": str(video_path), "nframes": num_frames}
