"""Tests for the memory planner."""

from voxscriber.memory import (
    PARALLEL_GB_PER_SECOND,
    SAFETY_MARGIN,
    SEQUENTIAL_GB_PER_SECOND,
    MemoryPlan,
    plan,
)


def test_parallel_fits_when_plenty_of_ram():
    # 1 hour audio on 48GB machine: trivially fits
    p = plan(duration_seconds=3600, requested_parallel=True, available_gb=48.0)
    assert p.mode == "parallel"
    assert isinstance(p, MemoryPlan)


def test_downgrades_to_sequential_when_parallel_too_big():
    # Recreate the original failure: ~3 hours on ~8GB free
    duration = 10765
    available = 8.0
    p = plan(duration_seconds=duration, requested_parallel=True, available_gb=available)
    # Parallel estimate exceeds budget, sequential estimate fits
    assert p.estimated_parallel_gb > available * SAFETY_MARGIN
    assert p.estimated_sequential_gb <= available * SAFETY_MARGIN
    assert p.mode == "sequential"
    assert "falling back to sequential" in p.reason


def test_skip_when_even_sequential_too_big():
    # 10 hours of audio on 2GB free
    p = plan(duration_seconds=36000, requested_parallel=True, available_gb=2.0)
    assert p.mode == "skip"
    assert "too long" in p.reason


def test_respects_user_requested_sequential():
    # User asked for sequential; we should stay sequential even if parallel would fit
    p = plan(duration_seconds=600, requested_parallel=False, available_gb=48.0)
    assert p.mode == "sequential"


def test_no_psutil_falls_back_to_requested_mode():
    # available_gb=None via argument simulates psutil missing
    # Pass a sentinel that the function treats as "unknown"
    import voxscriber.memory as mem

    original = mem._available_gb
    try:
        mem._available_gb = lambda: None
        p = mem.plan(duration_seconds=600, requested_parallel=True)
        assert p.mode == "parallel"
        assert "unavailable" in p.reason.lower()
    finally:
        mem._available_gb = original


def test_calibration_constants_ordered_correctly():
    # Sequential should always be cheaper than parallel, otherwise the planner is useless
    assert SEQUENTIAL_GB_PER_SECOND < PARALLEL_GB_PER_SECOND
    assert 0 < SAFETY_MARGIN <= 1.0
