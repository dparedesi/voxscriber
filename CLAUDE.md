# CLAUDE.md - VoxScriber

## Project Overview

VoxScriber is a speaker diarization CLI tool for Apple Silicon Macs. It combines MLX Whisper (Apple Silicon optimized) with Pyannote Audio 3.1 to produce speaker-attributed transcripts from audio files.

**Key Features:**
- 100% local processing (no cloud APIs)
- Apple Silicon optimized via MLX
- Multiple output formats (md, txt, json, srt, vtt)
- Parallel transcription + diarization for speed

## Repository Structure

```
src/voxscriber/
├── cli.py          # CLI entry point, argument parsing
├── pipeline.py     # Main orchestrator (DiarizationPipeline, PipelineConfig)
├── transcriber.py  # MLX Whisper wrapper
├── diarizer.py     # Pyannote speaker diarization
├── aligner.py      # Word-speaker alignment + hallucination filter
├── preprocessor.py # Audio format conversion (ffmpeg)
├── formatters.py   # Output formatting (md, txt, json, srt, vtt)
├── __init__.py     # Public exports
└── __main__.py     # python -m voxscriber support
```

## Commands

```bash
# Development
make install        # pip install -e .
make dev            # pip install -e ".[dev]"
make lint           # ruff check + format check
make format         # ruff format

# Publishing (requires ~/.pypirc with PyPI token)
# 1. Bump version in pyproject.toml
# 2. Commit and push
# 3. Create GitHub release: gh release create v0.x.x --title "v0.x.x" --notes "..."
# GitHub Actions handles PyPI upload automatically

# Testing locally
export HF_TOKEN=your_token
voxscriber test.m4a --speakers 2
```

## Architecture

1. **Preprocessor** converts input audio to 16kHz WAV
2. **Transcriber** (MLX Whisper) produces word-level timestamps
3. **Diarizer** (Pyannote) produces speaker segments
4. **Aligner** maps words to speakers, filters hallucinations
5. **Formatters** output to requested formats

Transcription and diarization run in parallel by default for speed.

## Key Dependencies

- `mlx-whisper` - Apple Silicon optimized Whisper
- `pyannote.audio` - Speaker diarization (requires HF token + terms acceptance)
- `pydub` + `soundfile` - Audio handling
- `rich` - Console output

## Environment Variables

- `HF_TOKEN` - Required. Hugging Face token for Pyannote models

## Coding Conventions

- Python 3.10+
- Ruff for linting/formatting (line-length: 100)
- Type hints encouraged
- Minimal dependencies
- No cloud APIs - everything runs locally

## Important Notes

- Pyannote requires users to accept terms at huggingface.co/pyannote/speaker-diarization-3.1
- Only runs on Apple Silicon (MLX requirement)
- Version must be bumped for any PyPI release (can't overwrite)
