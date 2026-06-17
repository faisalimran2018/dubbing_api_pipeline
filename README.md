# 🎙️ Auto-Dub API — WhisperX → NLLB-200 → XTTS v2

A single-endpoint FastAPI server that takes any audio file and returns a fully
dubbed Arabic version of it — automatically handling transcription, speaker
diarization, translation, and voice-cloned speech synthesis.

---

## Pipeline Overview

```
 ┌─────────────────────────────────────────────────────────────┐
 │                     POST /dub                               │
 │                                                             │
 │  Audio File                                                 │
 │      │                                                      │
 │      ▼                                                      │
 │  ① WhisperX large-v2                                        │
 │     • Transcribe speech to text (auto-detect language)      │
 │     • Word-level timestamp alignment                        │
 │     • Speaker diarization  (who spoke when)                 │
 │      │                                                      │
 │      ▼  segments: [{speaker, text, start, end}, …]          │
 │                                                             │
 │  ② NLLB-200 (facebook/nllb-200-distilled-1.3B)             │
 │     • Translate each segment to Modern Standard Arabic      │
 │     • Auto-detect source language from WhisperX output      │
 │      │                                                      │
 │      ▼  arabic_text per segment                             │
 │                                                             │
 │  ③ XTTS v2  (Coqui TTS)                                     │
 │     • Extract a reference voice clip per speaker            │
 │     • Synthesise Arabic speech in that speaker's voice      │
 │     • Time-stretch via ffmpeg atempo to fit original timing │
 │      │                                                      │
 │      ▼                                                      │
 │  ④ pydub assembly                                           │
 │     • Overlay each segment at its original timestamp        │
 │     • Export final dubbed WAV                               │
 └─────────────────────────────────────────────────────────────┘
```

---

## Models Used

| Stage | Model | Notes |
|-------|-------|-------|
| Transcription | **WhisperX large-v2** | OpenAI Whisper via faster-whisper backend |
| Diarization | **pyannote/speaker-diarization-3.1** | Pulled in by WhisperX; needs HF token |
| Translation | **NLLB-200 distilled 1.3B** | Meta's No Language Left Behind; 200 languages |
| TTS / Voice Cloning | **XTTS v2** | Coqui; 17 languages; 6-second voice clone |

> **Translation model answer:** The repo uses **NLLB-200** (`facebook/nllb-200-distilled-1.3B`), an open-source sequence-to-sequence model from Meta AI that supports translation between 200 languages without needing a pivot through English.

---

## Requirements

### System dependencies
- **Python 3.9–3.11**
- **ffmpeg** + **ffprobe** on PATH

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org and add bin/ to PATH
```

### HuggingFace account
WhisperX uses `pyannote/speaker-diarization-3.1` which requires:
1. A free HF account at https://huggingface.co
2. Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Create an access token at https://huggingface.co/settings/tokens

You pass this token as `hf_token` in every `/dub` request.

---

## Installation

### 1. Create virtual environment

```bash
python -m venv .venv
# Linux / Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2. Install PyTorch (GPU strongly recommended)

```bash
# CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (slow)
pip install torch torchaudio
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Accept Coqui TTS license

On first run XTTS v2 will ask you to accept the Coqui CPML license.
Skip the interactive prompt by setting:

```bash
# Linux / Mac
export COQUI_TOS_AGREED=1

# Windows
set COQUI_TOS_AGREED=1
```

---

## Running the Server

```bash
uvicorn API_python:app --host 0.0.0.0 --port 8000
```

- **Swagger UI:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

> On first start the server downloads ~4 GB of models (WhisperX large-v2 +
> NLLB-200 1.3B + XTTS v2). Subsequent starts are instant.

---

## API Reference

### `POST /dub`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file | ✅ | Source audio (WAV, MP3, M4A, FLAC, …) |
| `hf_token` | string (form) | ✅ | HuggingFace access token for diarization |
| `language` | string (form) | ❌ | ISO-639-1 hint e.g. `ur`, `en`. Omit for auto-detect. |

**Response:** `audio/wav` — the dubbed audio file.

---

### `GET /health`

Returns model status and device info.

```json
{
  "status": "ok",
  "device": "cuda",
  "whisperx_model": "large-v2",
  "translation_model": "facebook/nllb-200-distilled-1.3B",
  "tts_model": "xtts_v2",
  "target_language": "arb_Arab"
}
```

---

## Usage Examples

### curl

```bash
curl -X POST http://localhost:8000/dub \
  -F "audio=@my_audio.wav" \
  -F "hf_token=hf_xxxxxxxxxxxxxxxxxxxx" \
  -F "language=ur" \
  --output dubbed_output.wav
```

### Python (requests)

```python
import requests

with open("my_audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/dub",
        files={"audio": ("audio.wav", f, "audio/wav")},
        data={
            "hf_token": "hf_xxxxxxxxxxxxxxxxxxxx",
            "language": "ur",   # optional — omit for auto-detect
        },
    )

with open("dubbed_output.wav", "wb") as out:
    out.write(response.content)

print("Saved dubbed_output.wav")
```

---

## Project Structure

```
.
├── API_python.py          # Main FastAPI app — full pipeline
├── requirements.txt       # Python dependencies
├── outputs/               # Dubbed WAV files (auto-created, one per request)
├── temp/                  # Intermediate files, cleaned after each request
├── speakers/              # Optional: pre-supplied reference WAVs
└── README.md
```

---

## How Speaker Voice Cloning Works

The API does **not** require you to upload separate reference WAVs.
Instead, for each speaker detected by WhisperX it automatically:

1. Finds that speaker's **longest diarized segment** in the original audio
2. Extracts it as a reference clip (clamped to 6–30 seconds for XTTS)
3. Uses that clip as the voice reference for all of that speaker's dubbed segments

This means the cloned voice is always from the same audio you're dubbing —
no extra files needed.

---

## Configuration

Edit the constants near the top of `API_python.py`:

| Constant | Default | Notes |
|----------|---------|-------|
| `WHISPERX_MODEL` | `"large-v2"` | Use `"medium"` to save VRAM |
| `NLLB_MODEL_ID` | `"facebook/nllb-200-distilled-1.3B"` | Swap for `"facebook/nllb-200-3.3B"` for best quality (~12 GB VRAM) |
| `TARGET_LANG_NLLB` | `"arb_Arab"` | Modern Standard Arabic |
| `COMPUTE_TYPE` | `"float16"` (GPU) / `"int8"` (CPU) | Whisper compute precision |

---

## VRAM Requirements

| Config | Approx VRAM |
|--------|-------------|
| WhisperX large-v2 | ~5 GB |
| NLLB-200 1.3B | ~3 GB |
| XTTS v2 | ~3 GB |
| **Total (all loaded)** | **~11 GB** |

For 8 GB VRAM: switch to `"medium"` for WhisperX and `"facebook/nllb-200-distilled-600M"` for translation.
For CPU: it works but expect 10–30× slower processing.

---

## Troubleshooting

**`ffmpeg not found`** — Install ffmpeg and confirm with `ffmpeg -version` in the same shell.

**`OSError: 401 Unauthorized` from pyannote** — Your HF token is missing or hasn't been granted access to the diarization model. Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and accept the terms.

**CUDA out of memory** — Lower `WHISPERX_MODEL` to `"medium"` and/or switch NLLB to the 600M variant.

**Audio sounds too fast/slow** — This is the `atempo` stretch. For very short segments (<1 s), XTTS output can be much longer than the target window. Consider filtering out very short segments in the payload.

**Some segments skipped** — If a segment's speaker has no usable reference clip (e.g. under 1 second), it is skipped gracefully and logged. The rest of the audio is still assembled.

---

## License

Add your preferred license (MIT / Apache-2.0 / etc.).
