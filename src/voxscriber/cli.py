#!/usr/bin/env python3
"""VoxScriber CLI - Speaker diarization for Apple Silicon."""

import argparse
import os
import sys
from pathlib import Path

# Apply macOS fixes before any other imports
from .fix_path import setup_macos_environment
setup_macos_environment()

from dotenv import load_dotenv

from .pipeline import DiarizationPipeline, PipelineConfig

load_dotenv()


def run_diagnostics():
    """Run comprehensive diagnostics to help troubleshoot installation issues."""
    import platform
    import shutil
    import subprocess

    print("=" * 70)
    print("VoxScriber Diagnostics Report")
    print("=" * 70)
    print()

    # System info
    print("[System Information]")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Machine: {platform.machine()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  Python Path: {sys.executable}")
    print()

    # Check if running in venv
    print("[Python Environment]")
    in_venv = sys.prefix != sys.base_prefix
    print(f"  In Virtual Environment: {in_venv}")
    print(f"  sys.prefix: {sys.prefix}")
    if in_venv:
        print(f"  sys.base_prefix: {sys.base_prefix}")
    print()

    # Environment variables
    print("[Environment Variables]")
    for var in ["HF_TOKEN", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH"]:
        val = os.environ.get(var)
        if var == "HF_TOKEN" and val:
            print(f"  {var}: [SET - {len(val)} chars]")
        else:
            print(f"  {var}: {val or '[NOT SET]'}")
    print()

    # FFmpeg check
    print("[FFmpeg]")
    ffmpeg_path = shutil.which("ffmpeg")
    print(f"  ffmpeg binary: {ffmpeg_path or '[NOT FOUND]'}")
    if ffmpeg_path:
        try:
            result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True)
            version_line = result.stdout.split("\n")[0] if result.stdout else "Unknown"
            print(f"  Version: {version_line}")
        except Exception as e:
            print(f"  Version check failed: {e}")

    # Check for FFmpeg libraries
    lib_paths = ["/opt/homebrew/lib", "/usr/local/lib", "/usr/lib"]
    print(f"  Library search paths: {lib_paths}")
    for path in lib_paths:
        if os.path.isdir(path):
            avutil_libs = [f for f in os.listdir(path) if f.startswith("libavutil")]
            if avutil_libs:
                print(f"  Found in {path}: {', '.join(sorted(avutil_libs)[:3])}")
    print()

    # Package versions
    print("[Installed Packages]")
    packages = [
        "voxscriber", "pyannote-audio", "torch", "torchaudio",
        "torchcodec", "mlx", "mlx-whisper", "soundfile", "pydub"
    ]
    for pkg in packages:
        try:
            from importlib.metadata import version
            ver = version(pkg.replace("-", "_").replace("pyannote_audio", "pyannote-audio"))
        except Exception:
            try:
                ver = version(pkg)
            except Exception:
                ver = "[NOT INSTALLED]"
        print(f"  {pkg}: {ver}")
    print()

    # Test torch import
    print("[PyTorch Loading]")
    try:
        import torch
        print("  torch imported: OK")
        print(f"  torch version: {torch.__version__}")
        print(f"  torch location: {torch.__file__}")
        print(f"  MPS available: {torch.backends.mps.is_available()}")
    except Exception as e:
        print(f"  torch import FAILED: {e}")
    print()

    # Test torchcodec (the likely problem area)
    print("[TorchCodec Loading]")
    try:
        import torchcodec
        print("  torchcodec imported: OK")
        print(f"  torchcodec location: {torchcodec.__file__}")
        try:
            import torchcodec._core as core
            ffmpeg_versions = core.get_ffmpeg_library_versions()
            print(f"  FFmpeg versions detected: {ffmpeg_versions}")
        except Exception as e:
            print(f"  FFmpeg detection FAILED: {e}")
        try:
            from torchcodec.decoders import AudioDecoder  # noqa: F401
            print("  AudioDecoder import: OK")
        except Exception as e:
            print(f"  AudioDecoder import FAILED: {e}")
    except ImportError:
        print("  torchcodec: [NOT INSTALLED]")
    except Exception as e:
        print(f"  torchcodec import FAILED: {e}")
        import traceback
        print("  --- Traceback ---")
        for line in traceback.format_exc().split("\n"):
            print(f"  {line}")
        print("  --- End Traceback ---")
    print()

    # Test torchaudio
    print("[TorchAudio Loading]")
    try:
        import torchaudio
        print("  torchaudio imported: OK")
        print(f"  torchaudio version: {torchaudio.__version__}")
        # Check for AudioMetaData (removed in 2.9+)
        if hasattr(torchaudio, "AudioMetaData"):
            print("  torchaudio.AudioMetaData: EXISTS")
        else:
            print("  torchaudio.AudioMetaData: MISSING (removed in torchaudio 2.9+)")
    except Exception as e:
        print(f"  torchaudio import FAILED: {e}")
    print()

    # Test pyannote
    print("[Pyannote Loading]")
    try:
        from pyannote.audio import Pipeline  # noqa: F401
        print("  pyannote.audio.Pipeline imported: OK")
    except Exception as e:
        print(f"  pyannote.audio import FAILED: {e}")
        import traceback
        print("  --- Traceback ---")
        for line in traceback.format_exc().split("\n"):
            print(f"  {line}")
        print("  --- End Traceback ---")
    print()

    # Test MLX
    print("[MLX Loading]")
    try:
        import mlx.core  # noqa: F401
        print("  mlx imported: OK")
    except Exception as e:
        print(f"  mlx import FAILED: {e}")

    try:
        import mlx_whisper  # noqa: F401
        print("  mlx_whisper imported: OK")
    except Exception as e:
        print(f"  mlx_whisper import FAILED: {e}")
    print()

    # Library path resolution test
    print("[Library Path Resolution]")
    try:
        import ctypes.util
        for lib in ["avutil", "avcodec", "avformat"]:
            found = ctypes.util.find_library(lib)
            print(f"  find_library('{lib}'): {found or '[NOT FOUND]'}")
    except Exception as e:
        print(f"  Library resolution test failed: {e}")
    print()

    print("=" * 70)
    print("End of Diagnostics Report")
    print("=" * 70)
    print()
    print("Please share this output when reporting issues.")


def main():
    parser = argparse.ArgumentParser(
        description="VoxScriber - Speaker diarization with MLX Whisper + Pyannote",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    voxscriber meeting.m4a
    voxscriber meeting.m4a --speakers 2
    voxscriber meeting.m4a --formats md,txt,json

Environment:
    HF_TOKEN    Hugging Face token for pyannote models (required)
        """
    )

    parser.add_argument("audio", nargs="?", type=Path, help="Path to audio file")
    parser.add_argument("--output", "-o", type=Path, help="Output directory")
    parser.add_argument("--formats", "-f", type=str, default="md,txt",
                        help="Output formats: md,txt,json,srt,vtt (default: md,txt)")
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
    parser.add_argument("--diagnose", action="store_true",
                        help="Run diagnostics and print environment info (for troubleshooting)")

    args = parser.parse_args()

    if args.diagnose:
        run_diagnostics()
        sys.exit(0)

    if not args.audio:
        parser.error("the following arguments are required: audio")

    if not args.audio.exists():
        print(f"Error: File not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("""
Error: Hugging Face token required.

1. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
2. Get token at https://huggingface.co/settings/tokens
3. export HF_TOKEN=your_token_here
""", file=sys.stderr)
        sys.exit(1)

    config = PipelineConfig(
        whisper_model=args.model,
        language=args.language,
        hf_token=hf_token,
        num_speakers=args.speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        device=args.device,
        parallel=not args.sequential,
        verbose=not args.quiet,
    )

    pipeline = DiarizationPipeline(config)

    try:
        transcript = pipeline.process(
            audio_path=args.audio,
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
        print(f"Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
