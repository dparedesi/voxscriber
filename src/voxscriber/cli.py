#!/usr/bin/env python3
"""VoxScriber CLI - Speaker diarization for Apple Silicon."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .pipeline import DiarizationPipeline, PipelineConfig

load_dotenv()


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

    parser.add_argument("audio", type=Path, help="Path to audio file")
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

    args = parser.parse_args()

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
