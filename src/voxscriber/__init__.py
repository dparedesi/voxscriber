"""VoxScriber - Local speaker diarization for Apple Silicon."""

import platform
import os
import ctypes
from pathlib import Path

# MAC OS FIX: Pre-load FFmpeg libraries for torchcodec
# torchcodec often fails to find FFmpeg libraries on macOS due to RPATH issues.
# We explicitly load them from Homebrew paths to ensure they are available.
if platform.system() == "Darwin":
    def _preload_ffmpeg():
        # Common locations for Homebrew on Apple Silicon and Intel
        search_paths = [
            "/opt/homebrew/lib",
            "/usr/local/lib",
        ]

        # Libraries torchcodec depends on (order matters: util -> codec -> format)
        libs = ["libavutil", "libavcodec", "libavformat"]

        for lib_dir in search_paths:
            if not os.path.exists(lib_dir):
                continue

            # Try to load libraries from this directory
            loaded_count = 0
            for lib_name in libs:
                # Try versioned names first (e.g., libavutil.59.dylib) then generic
                # We specifically look for the .dylib versions
                try:
                    # Construct full path. We search for standard patterns.
                    # This is a bit heuristic, but effective.
                    found = False
                    for filename in os.listdir(lib_dir):
                        if filename.startswith(lib_name) and filename.endswith(".dylib"):
                            # Prefer loading the versioned dylib (e.g. libavutil.59.dylib)
                            # Avoiding simple symlinks if possible might be safer, but loading the symlink is usually fine
                            full_path = os.path.join(lib_dir, filename)
                            try:
                                ctypes.CDLL(full_path)
                                found = True
                                break
                            except OSError:
                                continue

                    if found:
                        loaded_count += 1
                except Exception:
                    pass

            # If we loaded all 3, we're probably good
            if loaded_count == 3:
                break

    _preload_ffmpeg()

from .pipeline import DiarizationPipeline, PipelineConfig

__all__ = ["DiarizationPipeline", "PipelineConfig"]
__version__ = "0.1.8"
