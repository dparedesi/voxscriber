# VoxScriber

[![PyPI version](https://img.shields.io/pypi/v/voxscriber.svg)](https://pypi.org/project/voxscriber/)
[![Downloads](https://pepy.tech/badge/voxscriber)](https://pepy.tech/project/voxscriber)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

<a href="https://buymeacoffee.com/dparedesi" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

Professional speaker diarization running 100% locally on Apple Silicon. Combines [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) with [Pyannote 3.1](https://github.com/pyannote/pyannote-audio).

![VoxScriber Banner](images/banner.png)

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- FFmpeg 7 (`brew install ffmpeg@7 && brew link ffmpeg@7`)
- [Hugging Face token](https://huggingface.co/settings/tokens) (free, one-time model download)

## Installation

```bash
# From PyPI
pip install voxscriber

# Or with pipx (recommended for CLI tools)
pipx install voxscriber
```

### Setup Hugging Face Token

VoxScriber uses pyannote models which require a Hugging Face token.

**Option 1: Interactive setup (recommended)**

```bash
voxscriber-doctor
```

This will guide you through accepting the model terms and saving your token securely.

**Option 2: Using huggingface-cli**

```bash
# First, accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
huggingface-cli login
```

Your token will be saved to `~/.cache/huggingface/token` and used automatically.

**Option 3: Environment variable**

```bash
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

# Sentence-level subtitle segmentation for editing workflows
voxscriber meeting.m4a --formats srt,vtt --srt-mode sentence --srt-max-duration 15

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
  --srt-mode        Subtitle segmentation mode for srt/vtt: speaker|sentence
  --srt-max-duration  Maximum subtitle duration in seconds for srt/vtt
  --quiet, -q       Suppress progress
  --print           Print transcript to console
```

## Performance

~0.1-0.15x RTF on Apple Silicon. A 20-minute recording processes in ~2-3 minutes.

## Troubleshooting

Run the diagnostic tool to check your setup:

```bash
voxscriber-doctor
```

This will check FFmpeg, torchcodec, and HF_TOKEN, and offer to fix common issues automatically.

### FFmpeg & torchcodec Issues

VoxScriber uses pyannote-audio which requires torchcodec, and torchcodec requires FFmpeg 4-7.

**"FFmpeg 8 detected" or "torchcodec fails"**

FFmpeg 8 is not yet supported. Install FFmpeg 7:

```bash
brew uninstall ffmpeg
brew install ffmpeg@7 && brew link ffmpeg@7
```

**"Library not loaded: @rpath/libavutil" or "no LC_RPATH's found"**

This happens because `ffmpeg@7` is "keg-only" - Homebrew doesn't symlink it automatically. Add to your `~/.zshrc`:

```bash
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib:$DYLD_LIBRARY_PATH"
```

Then restart your terminal or run `source ~/.zshrc`.

### Other Issues

| Issue | Solution |
|-------|----------|
| `requires Python >= 3.10` | Use Python 3.10+: `python3.10 -m venv .venv` |
| Installed wrong package | It's `voxscriber` (with 'r'), not `voxscribe` |
| `HF_TOKEN required` | Run `voxscriber-doctor` to set up authentication |

## Support

If you find VoxScriber useful, consider supporting its development:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-donate-yellow.svg)](https://buymeacoffee.com/dparedesi)
[![GitHub Sponsors](https://img.shields.io/badge/GitHub%20Sponsors-sponsor-pink.svg)](https://github.com/sponsors/dparedesi)

## License

MIT
