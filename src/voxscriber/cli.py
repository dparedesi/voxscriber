#!/usr/bin/env python3
"""VoxScriber CLI - Speaker diarization for Apple Silicon."""

import argparse
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

from .pipeline import DiarizationPipeline, PipelineConfig

load_dotenv()

# FFmpeg library path for keg-only Homebrew installations
FFMPEG7_LIB_PATH = "/opt/homebrew/opt/ffmpeg@7/lib"

# Pyannote model that requires acceptance of terms
PYANNOTE_MODEL_URL = "https://huggingface.co/pyannote/speaker-diarization-3.1"

# Supported audio file extensions (formats ffmpeg can handle)
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".webm"}


def _find_audio_files(directory: Path) -> List[Path]:
    """
    Find all audio files in a directory (non-recursive).

    Args:
        directory: Directory to search

    Returns:
        List of audio file paths, sorted alphabetically
    """
    audio_files = []
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files.append(file)
    return sorted(audio_files)


def _get_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """
    Get Hugging Face token from multiple sources in priority order.

    Priority:
    1. CLI argument (--hf-token)
    2. Environment variable (HF_TOKEN)
    3. huggingface_hub stored token (~/.cache/huggingface/token)

    Returns:
        Token string if found, None otherwise.
    """
    # 1. CLI argument takes highest priority
    if cli_token:
        return cli_token

    # 2. Environment variable
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token

    # 3. huggingface_hub stored token
    try:
        from huggingface_hub import get_token
        stored_token = get_token()
        if stored_token:
            return stored_token
    except ImportError:
        pass  # huggingface_hub not available
    except Exception:
        pass  # Any other error

    return None


def _get_hf_token_source(cli_token: Optional[str] = None) -> Tuple[Optional[str], str]:
    """
    Get Hugging Face token and its source.

    Returns:
        Tuple of (token, source_description)
    """
    if cli_token:
        return cli_token, "CLI argument (--hf-token)"

    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token, "environment variable (HF_TOKEN)"

    try:
        from huggingface_hub import get_token
        stored_token = get_token()
        if stored_token:
            return stored_token, "huggingface-cli login (~/.cache/huggingface/token)"
    except ImportError:
        pass
    except Exception:
        pass

    return None, "not found"


def _validate_hf_token(token: str) -> Tuple[bool, str]:
    """
    Validate that an HF token is valid by calling the API.

    Returns:
        Tuple of (is_valid, username_or_error)
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami(token=token)
        return True, user_info.get("name", "unknown")
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Invalid" in error_msg:
            return False, "Token is invalid or expired"
        return False, f"Could not validate: {error_msg[:100]}"


def _run_hf_login() -> bool:
    """
    Run interactive Hugging Face login.

    Returns:
        True if login was successful, False otherwise.
    """
    try:
        from huggingface_hub import login
        print()
        print("  Starting Hugging Face login...")
        print("  (Your token will be saved securely to ~/.cache/huggingface/token)")
        print()
        login(add_to_git_credential=False)
        return True
    except ImportError:
        print("  Error: huggingface_hub not installed")
        return False
    except Exception as e:
        print(f"  Login failed: {e}")
        return False


def _get_ffmpeg_info() -> Tuple[Optional[str], Optional[int]]:
    """Get FFmpeg path and major version."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return None, None

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True
        )
        version_match = re.search(r"ffmpeg version (\d+)", result.stdout)
        if version_match:
            return ffmpeg_path, int(version_match.group(1))
    except Exception:
        pass

    return ffmpeg_path, None


def _check_torchcodec_native_lib() -> Tuple[bool, str]:
    """
    Check if torchcodec native library can load.

    Returns:
        Tuple of (success, error_message)
    """
    try:
        from torchcodec.decoders import AudioDecoder  # noqa: F401
        return True, ""
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def _is_ffmpeg7_keg_only() -> bool:
    """Check if ffmpeg@7 is installed as keg-only (not symlinked)."""
    return os.path.isdir(FFMPEG7_LIB_PATH)


def _is_dyld_library_path_set() -> bool:
    """Check if DYLD_LIBRARY_PATH includes ffmpeg@7 lib path."""
    dyld_path = os.environ.get("DYLD_LIBRARY_PATH", "")
    return FFMPEG7_LIB_PATH in dyld_path


def check_dependencies() -> list[str]:
    """
    Check system dependencies and return list of errors.

    Validates:
    1. FFmpeg is installed and version is 4-7
    2. torchcodec native library can load FFmpeg
    3. DYLD_LIBRARY_PATH is set if using keg-only ffmpeg@7
    """
    errors = []

    # Step 1: Check FFmpeg installation and version
    ffmpeg_path, ffmpeg_version = _get_ffmpeg_info()

    if not ffmpeg_path:
        errors.append(
            "FFmpeg not found.\n"
            "  Fix: brew install ffmpeg@7 && brew link ffmpeg@7"
        )
        return errors  # Can't proceed without FFmpeg

    if ffmpeg_version is not None:
        if ffmpeg_version > 7:
            errors.append(
                f"FFmpeg {ffmpeg_version} detected, but version 4-7 is required.\n"
                "  torchcodec (used by pyannote-audio) does not yet support FFmpeg 8.\n"
                "  Fix: brew uninstall ffmpeg && brew install ffmpeg@7 && brew link ffmpeg@7"
            )
            return errors  # Wrong version, can't proceed
        elif ffmpeg_version < 4:
            errors.append(
                f"FFmpeg {ffmpeg_version} is too old. Version 4-7 required.\n"
                "  Fix: brew install ffmpeg@7 && brew link ffmpeg@7"
            )
            return errors  # Wrong version, can't proceed

    # Step 2: Check if torchcodec can load its native library
    torchcodec_ok, torchcodec_error = _check_torchcodec_native_lib()

    if torchcodec_ok:
        return []  # All good!

    # Step 3: torchcodec failed - diagnose the issue
    is_library_path_error = any(x in torchcodec_error for x in [
        "libavutil", "libtorchcodec", "Library not loaded", "no LC_RPATH"
    ])

    if is_library_path_error:
        # Check if this is the keg-only ffmpeg@7 issue
        if _is_ffmpeg7_keg_only() and not _is_dyld_library_path_set():
            errors.append(
                "torchcodec cannot find FFmpeg libraries.\n\n"
                "  This happens because ffmpeg@7 is 'keg-only' - Homebrew doesn't\n"
                "  symlink it to /opt/homebrew/lib automatically.\n\n"
                "  Fix: Add this to your ~/.zshrc (or ~/.bashrc):\n\n"
                f'    export DYLD_LIBRARY_PATH="{FFMPEG7_LIB_PATH}:$DYLD_LIBRARY_PATH"\n\n'
                "  Then restart your terminal or run: source ~/.zshrc"
            )
        else:
            # Generic library loading error
            errors.append(
                "torchcodec cannot load FFmpeg libraries.\n\n"
                "  Possible fixes:\n"
                "  1. Reinstall FFmpeg: brew reinstall ffmpeg@7\n"
                "  2. Ensure FFmpeg 7 is linked: brew link ffmpeg@7\n"
                "  3. If using ffmpeg@7, add to ~/.zshrc:\n"
                f'     export DYLD_LIBRARY_PATH="{FFMPEG7_LIB_PATH}:$DYLD_LIBRARY_PATH"'
            )
    else:
        # Some other torchcodec import error
        errors.append(f"torchcodec import error: {torchcodec_error}")

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="VoxScriber - Speaker diarization with MLX Whisper + Pyannote",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  voxscriber meeting.m4a                    # Basic usage
  voxscriber meeting.m4a --speakers 2       # Known speaker count
  voxscriber meeting.m4a --formats md,json  # Multiple output formats
  voxscriber ./recordings/                  # Batch process a folder

Output Formats:
  md    Markdown with bold speaker names
  txt   Timestamped plain text
  json  Structured data with word-level timestamps
  srt   SubRip subtitles
  vtt   WebVTT subtitles

First-Time Setup:
  1. Accept model terms at: {PYANNOTE_MODEL_URL}
  2. Run: voxscriber-doctor   (interactive setup wizard)
     Or manually: huggingface-cli login

Troubleshooting:
  Run 'voxscriber-doctor' to diagnose and fix common issues.
"""
    )

    parser.add_argument("audio", type=Path, help="Path to audio file or folder")
    parser.add_argument("--output", "-o", type=Path, help="Output directory")
    parser.add_argument("--formats", "-f", type=str, default="md",
                        help="Output formats: md,txt,json,srt,vtt (default: md)")
    parser.add_argument("--model", "-m", type=str, default="large-v3-turbo",
                        choices=["tiny", "base", "small", "medium", "large",
                                 "large-v3-turbo", "large-4bit", "large-8bit"],
                        help="Whisper model (default: large-v3-turbo)")
    parser.add_argument("--language", "-l", type=str, help="Force language (e.g., 'en', 'es')")
    parser.add_argument("--speakers", "-s", type=int, help="Number of speakers (if known)")
    parser.add_argument("--min-speakers", type=int, help="Minimum speakers")
    parser.add_argument("--max-speakers", type=int, help="Maximum speakers")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"],
                        help="Device (default: mps)")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")
    parser.add_argument("--sequential", action="store_true",
                        help="Run sequentially instead of parallel")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    parser.add_argument("--print", action="store_true", dest="print_result",
                        help="Print transcript to console")
    parser.add_argument("--srt-mode", type=str, default="speaker", choices=["speaker", "sentence"],
                        help="Subtitle segmentation mode for srt/vtt (default: speaker)")
    parser.add_argument("--srt-max-duration", type=float,
                        help="Maximum subtitle duration in seconds for srt/vtt (e.g., 15)")
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {importlib.metadata.version('voxscriber')}"
    )

    args = parser.parse_args()

    if args.srt_max_duration is not None and args.srt_max_duration <= 0:
        parser.error("--srt-max-duration must be > 0")

    # Check dependencies first
    dep_errors = check_dependencies()
    if dep_errors:
        print("Error: Dependency check failed:\n", file=sys.stderr)
        for err in dep_errors:
            print(f"  • {err}\n", file=sys.stderr)
        sys.exit(1)

    if not args.audio.exists():
        print(f"Error: Path not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # Determine if processing single file or batch
    is_batch = args.audio.is_dir()
    if is_batch:
        audio_files = _find_audio_files(args.audio)
        if not audio_files:
            print(f"Error: No audio files found in {args.audio}", file=sys.stderr)
            print(f"  Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}")
            sys.exit(1)
    else:
        audio_files = [args.audio]

    # Get HF token from multiple sources
    hf_token = _get_hf_token(args.hf_token)
    if not hf_token:
        # Check if running in interactive terminal
        if sys.stdin.isatty() and sys.stdout.isatty():
            print("\nNo Hugging Face token found.", file=sys.stderr)
            print("VoxScriber needs this to download speaker diarization models.\n")
            try:
                response = input("Run setup wizard (voxscriber-doctor)? [Y/n]: ").strip().lower()
                if response in ("", "y", "yes"):
                    print()
                    result = doctor()
                    if result == 0:
                        # Doctor succeeded, retry getting token
                        hf_token = _get_hf_token()
                        if hf_token:
                            print("\nSetup complete! Continuing with transcription...\n")
                        else:
                            print("\nSetup incomplete. Please run 'voxscriber-doctor' again.")
                            sys.exit(1)
                    else:
                        sys.exit(1)
                else:
                    # User declined, show manual instructions
                    print("""
To set up manually:
  1. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
  2. Run: huggingface-cli login
""")
                    sys.exit(1)
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                sys.exit(130)
        else:
            # Non-interactive: show full error message
            print("""
Error: Hugging Face token required.

VoxScriber needs a Hugging Face token to download pyannote models.

Quick setup (recommended):
    voxscriber-doctor

Manual setup:
    1. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
    2. Get token at https://huggingface.co/settings/tokens
    3. Run: huggingface-cli login
       (or set HF_TOKEN environment variable)
""", file=sys.stderr)
            sys.exit(1)

    # Process files
    total_files = len(audio_files)
    failed_files = []

    for idx, audio_file in enumerate(audio_files, 1):
        # Determine number of speakers for this file
        # If filename contains "1to1", it's a two-person conversation
        file_speakers = args.speakers
        if file_speakers is None and "1to1" in audio_file.stem.lower():
            file_speakers = 2

        if is_batch and not args.quiet:
            print(f"\n[{idx}/{total_files}] Processing: {audio_file.name}")
            if file_speakers == 2 and args.speakers is None:
                print("  (detected 1to1 in filename, setting speakers=2)")

        config = PipelineConfig(
            whisper_model=args.model,
            language=args.language,
            hf_token=hf_token,
            num_speakers=file_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            device=args.device,
            parallel=not args.sequential,
            verbose=not args.quiet,
            subtitle_mode=args.srt_mode,
            subtitle_max_duration=args.srt_max_duration,
        )

        pipeline = DiarizationPipeline(config)

        try:
            transcript = pipeline.process(
                audio_path=audio_file,
                output_dir=args.output,
                output_formats=[f.strip() for f in args.formats.split(",")],
            )

            if args.print_result:
                print("\n" + "=" * 60 + "\n")
                pipeline.print_transcript(transcript)

        except KeyboardInterrupt:
            print("\nInterrupted", file=sys.stderr)
            sys.exit(130)
        except Exception as e:
            if is_batch:
                failed_files.append((audio_file, str(e)))
                print(f"Error processing {audio_file.name}: {e}", file=sys.stderr)
            else:
                print(f"Error: {e}", file=sys.stderr)
                if not args.quiet:
                    import traceback
                    traceback.print_exc()
                sys.exit(1)

    # Summary for batch processing
    if is_batch and not args.quiet:
        print(f"\nBatch complete: {total_files - len(failed_files)}/{total_files} files processed")
        if failed_files:
            print("Failed files:")
            for f, err in failed_files:
                print(f"  - {f.name}: {err}")


def _get_shell_config_file() -> Optional[Path]:
    """Detect the user's shell and return the appropriate config file."""
    shell = os.environ.get("SHELL", "")
    home = Path.home()

    if "zsh" in shell:
        return home / ".zshrc"
    elif "bash" in shell:
        # On macOS, .bash_profile is used for login shells
        bash_profile = home / ".bash_profile"
        bashrc = home / ".bashrc"
        if bash_profile.exists():
            return bash_profile
        return bashrc
    return None


def _check_shell_config_has_dyld_path(config_file: Path) -> bool:
    """Check if the shell config already has the ffmpeg@7 DYLD_LIBRARY_PATH export."""
    if not config_file.exists():
        return False
    content = config_file.read_text()
    # Check specifically for the ffmpeg@7 lib path in a DYLD_LIBRARY_PATH export
    # This handles variations like:
    #   export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib:$DYLD_LIBRARY_PATH"
    #   export DYLD_LIBRARY_PATH='/opt/homebrew/opt/ffmpeg@7/lib:...'
    return FFMPEG7_LIB_PATH in content


def doctor():
    """
    Interactive diagnostic and setup tool for VoxScriber.

    Checks system dependencies and offers to fix common issues.
    """
    print("VoxScriber Doctor")
    print("=" * 40)
    print()

    all_ok = True

    # Check 1: FFmpeg
    print("Checking FFmpeg...", end=" ")
    ffmpeg_path, ffmpeg_version = _get_ffmpeg_info()

    if not ffmpeg_path:
        print("NOT FOUND")
        print("  FFmpeg is not installed.")
        print("  Fix: brew install ffmpeg@7 && brew link ffmpeg@7")
        all_ok = False
    elif ffmpeg_version is None:
        print("UNKNOWN VERSION")
        print(f"  FFmpeg found at {ffmpeg_path} but couldn't determine version.")
        all_ok = False
    elif ffmpeg_version > 7:
        print(f"VERSION {ffmpeg_version} (unsupported)")
        print("  torchcodec requires FFmpeg 4-7. Version 8 is not yet supported.")
        print("  Fix: brew uninstall ffmpeg && brew install ffmpeg@7 && brew link ffmpeg@7")
        all_ok = False
    elif ffmpeg_version < 4:
        print(f"VERSION {ffmpeg_version} (too old)")
        print("  torchcodec requires FFmpeg 4-7.")
        print("  Fix: brew install ffmpeg@7 && brew link ffmpeg@7")
        all_ok = False
    else:
        print(f"OK (version {ffmpeg_version})")

    # Check 2: torchcodec native library
    print("Checking torchcodec...", end=" ")
    torchcodec_ok, torchcodec_error = _check_torchcodec_native_lib()

    if torchcodec_ok:
        print("OK")
    else:
        print("FAILED")
        all_ok = False

        # Diagnose the issue
        is_library_path_error = any(x in torchcodec_error for x in [
            "libavutil", "libtorchcodec", "Library not loaded", "no LC_RPATH"
        ])

        if is_library_path_error and _is_ffmpeg7_keg_only():
            print()
            print("  torchcodec cannot find FFmpeg libraries.")
            print("  This is because ffmpeg@7 is 'keg-only' - Homebrew doesn't")
            print("  symlink it to /opt/homebrew/lib automatically.")
            print()

            # Check if we can offer to fix it
            shell_config = _get_shell_config_file()
            if shell_config:
                if _check_shell_config_has_dyld_path(shell_config):
                    print(f"  Your {shell_config.name} already has the DYLD_LIBRARY_PATH export.")
                    print("  Try restarting your terminal or running:")
                    print(f"    source {shell_config}")
                else:
                    print(f"  Would you like to add the fix to {shell_config}?")
                    print()
                    print("  This will append the following line:")
                    print(f'    export DYLD_LIBRARY_PATH="{FFMPEG7_LIB_PATH}:$DYLD_LIBRARY_PATH"')
                    print()

                    try:
                        response = input("  Add to shell config? [y/N]: ").strip().lower()
                        if response == "y":
                            export_line = (
                                f'\n# Added by voxscriber doctor for ffmpeg@7 support\n'
                                f'export DYLD_LIBRARY_PATH="{FFMPEG7_LIB_PATH}:$DYLD_LIBRARY_PATH"\n'
                            )
                            with open(shell_config, "a") as f:
                                f.write(export_line)
                            print()
                            print(f"  Added to {shell_config}!")
                            print()
                            print("  To apply the changes, run:")
                            print(f"    source {shell_config}")
                            print()
                            print("  Or restart your terminal.")
                        else:
                            print()
                            print("  Skipped. To fix manually, add this to your shell config:")
                            print(f'    export DYLD_LIBRARY_PATH="{FFMPEG7_LIB_PATH}:$DYLD_LIBRARY_PATH"')
                    except (EOFError, KeyboardInterrupt):
                        print("\n  Cancelled.")
            else:
                print("  Could not detect your shell config file.")
                print("  Add this to your shell config (~/.zshrc or ~/.bashrc):")
                print(f'    export DYLD_LIBRARY_PATH="{FFMPEG7_LIB_PATH}:$DYLD_LIBRARY_PATH"')
        else:
            print(f"  Error: {torchcodec_error[:200]}")

    # Check 3: HF Token
    print("Checking Hugging Face token...", end=" ")
    token, source = _get_hf_token_source()

    if token:
        # Mask the token for display
        masked = token[:8] + "..." if len(token) > 8 else "***"
        print(f"FOUND ({masked})")
        print(f"  Source: {source}")

        # Optionally validate the token
        print("  Validating token...", end=" ")
        is_valid, result = _validate_hf_token(token)
        if is_valid:
            print(f"OK (logged in as: {result})")
        else:
            print("FAILED")
            print(f"  {result}")
            all_ok = False
    else:
        print("NOT FOUND")
        all_ok = False
        print()
        print("  Hugging Face token is required for pyannote speaker diarization models.")
        print()
        print("  Before logging in, you must accept the model terms at:")
        print(f"    {PYANNOTE_MODEL_URL}")
        print()

        try:
            response = input("  Would you like to log in now? [y/N]: ").strip().lower()
            if response == "y":
                if _run_hf_login():
                    # Re-check if login was successful
                    new_token = _get_hf_token()
                    if new_token:
                        print()
                        print("  Login successful! Token saved.")
                        is_valid, result = _validate_hf_token(new_token)
                        if is_valid:
                            print(f"  Logged in as: {result}")
                            all_ok = True  # This check now passes
                    else:
                        print("  Login may have failed. Please try again.")
            else:
                print()
                print("  To set up authentication manually:")
                print(f"    1. Accept terms at: {PYANNOTE_MODEL_URL}")
                print("    2. Get token at: https://huggingface.co/settings/tokens")
                print("    3. Run: huggingface-cli login")
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")

    # Summary
    print()
    print("=" * 40)
    if all_ok:
        print("All checks passed! VoxScriber is ready to use.")
    else:
        print("Some issues were found. Please fix them and run 'voxscriber-doctor' again.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    main()
