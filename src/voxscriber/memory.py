"""
Memory planner.

Given an audio duration and the current system's available RAM, decide whether
we can safely run the pipeline in parallel mode, need to fall back to sequential,
or whether the file is likely to OOM at all.

This exists because running transcription + diarization in parallel on long audio
(a few hours) can blow past available RAM and get SIGKILL'd by the OS, which a
try/except cannot catch. A cheap upfront estimate avoids most of these crashes.

Calibration is rough and based on a single data point (a 10,765s file that OOM'd
on a 16GB Mac in parallel mode, while a 3,688s file on the same machine ran fine).
Refine `PARALLEL_GB_PER_SECOND` / `SEQUENTIAL_GB_PER_SECOND` as more data arrives.
"""

from dataclasses import dataclass
from typing import Literal, Optional

PARALLEL_GB_PER_SECOND = 0.001
SEQUENTIAL_GB_PER_SECOND = 0.0005
SAFETY_MARGIN = 0.7

Mode = Literal["parallel", "sequential", "skip"]


@dataclass
class MemoryPlan:
    mode: Mode
    reason: str
    estimated_parallel_gb: float
    estimated_sequential_gb: float
    available_gb: float


def _available_gb() -> Optional[float]:
    """Return available system RAM in GB, or None if psutil is unavailable."""
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        return None


def plan(
    duration_seconds: float,
    requested_parallel: bool = True,
    available_gb: Optional[float] = None,
) -> MemoryPlan:
    """
    Pick a processing mode for a file of the given duration.

    Args:
        duration_seconds: Audio duration in seconds.
        requested_parallel: What the user asked for (True = parallel, False = sequential).
        available_gb: Override for testing. If None, read from the system.

    Returns:
        MemoryPlan with the recommended mode and human-readable reason.
    """
    if available_gb is None:
        available_gb = _available_gb()

    est_parallel = duration_seconds * PARALLEL_GB_PER_SECOND
    est_sequential = duration_seconds * SEQUENTIAL_GB_PER_SECOND

    if available_gb is None:
        mode: Mode = "parallel" if requested_parallel else "sequential"
        return MemoryPlan(
            mode=mode,
            reason="memory check unavailable (psutil not installed), using requested mode",
            estimated_parallel_gb=est_parallel,
            estimated_sequential_gb=est_sequential,
            available_gb=0.0,
        )

    budget = available_gb * SAFETY_MARGIN

    if requested_parallel and est_parallel <= budget:
        return MemoryPlan(
            mode="parallel",
            reason=f"parallel mode fits: {est_parallel:.1f}GB needed, {budget:.1f}GB budget",
            estimated_parallel_gb=est_parallel,
            estimated_sequential_gb=est_sequential,
            available_gb=available_gb,
        )

    if est_sequential <= budget:
        reason = (
            f"falling back to sequential mode: parallel needs {est_parallel:.1f}GB, "
            f"only {budget:.1f}GB available"
            if requested_parallel
            else f"sequential mode fits: {est_sequential:.1f}GB needed, {budget:.1f}GB budget"
        )
        return MemoryPlan(
            mode="sequential",
            reason=reason,
            estimated_parallel_gb=est_parallel,
            estimated_sequential_gb=est_sequential,
            available_gb=available_gb,
        )

    return MemoryPlan(
        mode="skip",
        reason=(
            f"audio is too long for available memory: needs {est_sequential:.1f}GB "
            f"even in sequential mode, only {budget:.1f}GB available. "
            f"Split the file with ffmpeg and retry."
        ),
        estimated_parallel_gb=est_parallel,
        estimated_sequential_gb=est_sequential,
        available_gb=available_gb,
    )
