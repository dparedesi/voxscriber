"""
Fix for macOS library loading issues.

This module provides a function to automatically configure environment variables
and preload libraries to ensure torchcodec and other dependencies can find
native libraries (like FFmpeg) on macOS.
"""

import os
import sys
import platform
import ctypes
from pathlib import Path

def setup_macos_environment():
    """
    Configure macOS environment for FFmpeg library loading.

    This function:
    1. Identifies Homebrew library paths
    2. Updates DYLD_LIBRARY_PATH environment variable for the current process
    3. Pre-loads FFmpeg libraries using ctypes to ensure they are available

    This is necessary because torchcodec's native dylibs often use @rpath
    references that fail to resolve on macOS without explicit help, especially
    when installed in virtual environments.
    """
    if platform.system() != "Darwin":
        return

    # 1. Update DYLD_LIBRARY_PATH
    current_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")

    # Common locations for Homebrew on Apple Silicon and Intel
    homebrew_paths = ["/opt/homebrew/lib", "/usr/local/lib"]

    paths_to_add = []
    for path in homebrew_paths:
        if os.path.exists(path):
            # Only add if not already in path (to avoid duplication)
            if path not in current_dyld:
                paths_to_add.append(path)

    if paths_to_add:
        # Construct new path
        if current_dyld:
            new_path = ":".join(paths_to_add + [current_dyld])
        else:
            new_path = ":".join(paths_to_add)

        # Update environment variable for this process
        os.environ["DYLD_LIBRARY_PATH"] = new_path

        # NOTE: Updating os.environ inside the running process might not affect
        # the dynamic linker for libraries loaded *after* this point if the
        # process was started without the variable. However, it helps subprocesses
        # and some dlopen calls.

        # For torchcodec specifically, we need more aggressive measures (ctypes preloading)

    # 2. Pre-load FFmpeg libraries via ctypes
    # This is the most effective way to ensure symbols are available to other dylibs
    _preload_ffmpeg(homebrew_paths)


def _preload_ffmpeg(search_paths):
    """Explicitly load FFmpeg libraries using ctypes."""
    # Libraries torchcodec depends on (order matters: util -> codec -> format)
    libs = ["libavutil", "libavcodec", "libavformat"]

    for lib_dir in search_paths:
        if not os.path.exists(lib_dir):
            continue

        # Try to load libraries from this directory
        loaded_count = 0
        for lib_name in libs:
            try:
                # Find the dylib files
                found = False
                for filename in os.listdir(lib_dir):
                    if filename.startswith(lib_name) and filename.endswith(".dylib"):
                        full_path = os.path.join(lib_dir, filename)
                        try:
                            # RTLD_GLOBAL is important to make symbols available to other libraries
                            ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
                            found = True
                            break
                        except OSError:
                            continue

                if found:
                    loaded_count += 1
            except Exception:
                pass

        # If we loaded all 3 from this directory, we're likely good
        if loaded_count == 3:
            break
