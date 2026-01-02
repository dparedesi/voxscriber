# VoxScriber

[![PyPI version](https://img.shields.io/pypi/v/voxscriber.svg)](https://pypi.org/project/voxscriber/)
[![Downloads](https://pepy.tech/badge/voxscriber)](https://pepy.tech/project/voxscriber)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Professional speaker diarization running 100% locally on Apple Silicon. Combines [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) with [Pyannote 3.1](https://github.com/pyannote/pyannote-audio) for state-of-the-art results.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- FFmpeg (`brew install ffmpeg`)
- [Hugging Face token](https://huggingface.co/settings/tokens) (free, one-time model download)

## Installation

```bash
# From PyPI
pip install voxscriber

# Or with pipx (recommended for CLI tools)
pipx install voxscriber
```

### Setup Hugging Face Token

```bash
# Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
export HF_TOKEN=your_token_here
```

## Usage

```bash
# Basic
voxscriber meeting.m4a

# With known speaker count
voxscriber meeting.m4a --speakers 2

# All formats
voxscriber meeting.m4a --formats md,txt,json,srt,vtt

# Print to console
voxscriber meeting.m4a --print
```

### Python API

```python
from voxscriber import DiarizationPipeline, PipelineConfig

config = PipelineConfig(
    num_speakers=2,
    language="en",
)
pipeline = DiarizationPipeline(config)
transcript = pipeline.process("meeting.m4a")

for segment in transcript.segments:
    print(f"{segment.speaker}: {segment.text}")
```

## Output Formats

| Format | Description |
|--------|-------------|
| `md` | Markdown with bold speaker names |
| `txt` | Timestamped plain text |
| `json` | Structured data with word-level timestamps |
| `srt` | SubRip subtitles |
| `vtt` | WebVTT subtitles |

## Options

```
voxscriber --help

  --speakers, -s    Number of speakers (if known)
  --language, -l    Force language (e.g., 'en', 'es')
  --model, -m       Whisper model (default: large-v3-turbo)
  --formats, -f     Output formats (default: md,txt)
  --output, -o      Output directory
  --device          mps (default) or cpu
  --quiet, -q       Suppress progress
  --print           Print transcript to console
```

## Performance

~0.1-0.15x RTF on Apple Silicon. A 20-minute recording processes in ~2-3 minutes.

## Support

If you find VoxScriber useful, consider supporting its development:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-donate-yellow.svg)](https://buymeacoffee.com/dparedesi)
[![GitHub Sponsors](https://img.shields.io/badge/GitHub%20Sponsors-sponsor-pink.svg)](https://github.com/sponsors/dparedesi)

## License

MIT
