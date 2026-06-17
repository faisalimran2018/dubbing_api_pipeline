import os
import gc
import json
import uuid
import shutil
import tempfile
import subprocess
import traceback

import torch
import whisperx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydub import AudioSegment
from TTS.api import TTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ============================================
# CONFIG
# ============================================

OUTPUT_DIR = "outputs"
TEMP_DIR = "temp"
SPEAKERS_DIR = "speakers"          # reference WAVs for voice cloning
WHISPERX_MODEL = "large-v2"        # or "medium" for faster / lower VRAM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# NLLB-200 — distilled 1.3B is a good balance of quality vs. speed.
# Swap for "facebook/nllb-200-3.3B" for highest quality (needs ~12 GB VRAM).
NLLB_MODEL_ID = "facebook/nllb-200-distilled-1.3B"
TARGET_LANG_NLLB = "arb_Arab"      # Modern Standard Arabic in NLLB-200 codes

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(SPEAKERS_DIR, exist_ok=True)

# ============================================
# MODEL LOADING  (once at startup)
# ============================================

print(f"[startup] Device: {DEVICE}  |  Compute type: {COMPUTE_TYPE}")

# -- WhisperX transcription model --
print("[startup] Loading WhisperX ASR model …")
whisper_model = whisperx.load_model(
    WHISPERX_MODEL,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
)

# -- NLLB-200 translation --
print("[startup] Loading NLLB-200 translation model …")
nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID)
nllb_model = nllb_model.to(DEVICE)
nllb_model.eval()

# -- XTTS v2 TTS / voice cloning --
print("[startup] Loading XTTS v2 …")
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)

print("[startup] All models ready.")

app = FastAPI(title="Auto-Dub API  |  WhisperX → NLLB-200 → XTTS v2")


# ============================================
# HELPERS
# ============================================

def detect_language_nllb(text: str) -> str:
    """
    Ask NLLB-200's tokenizer to detect the dominant language of a short text
    sample and return the NLLB language code (e.g. 'urd_Arab', 'eng_Latn').

    WhisperX already gives us the ISO-639-1 code ('ur', 'en', …); we convert
    that to the NLLB code so the translation model knows the source language.
    """
    iso_to_nllb = {
        "ur": "urd_Arab",
        "en": "eng_Latn",
        "ar": "arb_Arab",
        "fr": "fra_Latn",
        "de": "deu_Latn",
        "es": "spa_Latn",
        "zh": "zho_Hans",
        "hi": "hin_Deva",
        "tr": "tur_Latn",
        "ru": "rus_Cyrl",
        "pt": "por_Latn",
        "it": "ita_Latn",
        "nl": "nld_Latn",
        "pl": "pol_Latn",
        "ko": "kor_Hang",
        "ja": "jpn_Jpan",
    }
    # Default to English if unmapped
    return iso_to_nllb.get(text.strip().lower(), "eng_Latn")


def translate_to_arabic(text: str, src_lang_nllb: str) -> str:
    """Translate *text* from *src_lang_nllb* to Arabic using NLLB-200."""
    if src_lang_nllb == TARGET_LANG_NLLB:
        return text  # already Arabic — skip translation

    inputs = nllb_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    target_lang_id = nllb_tokenizer.convert_tokens_to_ids(TARGET_LANG_NLLB)

    with torch.no_grad():
        generated = nllb_model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_new_tokens=512,
            num_beams=4,
        )

    translated = nllb_tokenizer.batch_decode(generated, skip_special_tokens=True)
    return translated[0]


def stretch_audio(input_path: str, output_path: str, target_duration: float):
    """
    Stretch / compress *input_path* to *target_duration* seconds via ffmpeg
    atempo (supports chaining for factors outside the 0.5–2.0 range).
    """
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    current_duration = float(
        subprocess.check_output(probe_cmd).decode().strip()
    )

    if current_duration <= 0:
        raise ValueError(f"ffprobe returned zero duration for {input_path}")

    speed = current_duration / target_duration

    filters = []
    s = speed
    while s > 2.0:
        filters.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        filters.append("atempo=0.5")
        s /= 0.5
    filters.append(f"atempo={s:.6f}")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter:a", ",".join(filters),
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def extract_speaker_reference(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    out_path: str,
    min_duration: float = 6.0,
    max_duration: float = 30.0,
):
    """
    Carve out a speaker reference clip from the original audio.
    Clamps to [min_duration, max_duration] seconds for XTTS.
    """
    duration = end_sec - start_sec
    duration = max(min_duration, min(duration, max_duration))

    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ss", str(start_sec),
        "-t", str(duration),
        "-ar", "22050",      # XTTS preferred sample rate
        "-ac", "1",          # mono
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def cleanup_temp_files(job_id: str):
    """Remove all temp files belonging to this job."""
    for f in os.listdir(TEMP_DIR):
        if f.startswith(job_id):
            try:
                os.remove(os.path.join(TEMP_DIR, f))
            except OSError:
                pass


# ============================================
# MAIN API
# ============================================

@app.post("/dub")
async def dub_audio(
    audio: UploadFile = File(..., description="Source audio file (WAV/MP3/etc.)"),
    hf_token: str = Form(..., description="HuggingFace token (needed for WhisperX diarization model)"),
    language: str = Form(default=None, description="ISO-639-1 source language hint, e.g. 'ur'. Leave blank for auto-detect."),
):
    """
    Full auto-dub pipeline:
      1. WhisperX  — transcribe + word-level alignment + speaker diarization
      2. NLLB-200  — translate each segment to Arabic
      3. XTTS v2   — synthesize Arabic speech in each speaker's cloned voice
      4. pydub     — assemble final audio, preserving original timing
    Returns a dubbed WAV file.
    """
    job_id = str(uuid.uuid4())
    original_audio_path = os.path.join(TEMP_DIR, f"{job_id}_original")

    # --- Save uploaded file (preserve original extension) ---
    ext = os.path.splitext(audio.filename)[-1].lower() or ".wav"
    original_audio_path += ext

    with open(original_audio_path, "wb") as f:
        f.write(await audio.read())

    try:
        # ------------------------------------------------
        # STEP 1 — WhisperX: transcribe + align + diarise
        # ------------------------------------------------
        print(f"[{job_id}] Step 1: WhisperX transcription …")
        audio_array = whisperx.load_audio(original_audio_path)

        # Transcribe
        wx_result = whisper_model.transcribe(
            audio_array,
            batch_size=16,
            language=language if language else None,   # None = auto-detect
        )
        detected_language = wx_result["language"]
        print(f"[{job_id}] Detected language: {detected_language}")

        # Word-level alignment
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=DEVICE,
        )
        wx_result = whisperx.align(
            wx_result["segments"],
            align_model,
            align_metadata,
            audio_array,
            DEVICE,
            return_char_alignments=False,
        )

        # Free alignment model VRAM before diarisation
        del align_model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Speaker diarization
        print(f"[{job_id}] Running speaker diarization …")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token,
            device=DEVICE,
        )
        diarize_segments = diarize_model(audio_array)
        wx_result = whisperx.assign_word_speakers(diarize_segments, wx_result)

        del diarize_model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        segments = wx_result["segments"]
        if not segments:
            raise HTTPException(status_code=422, detail="WhisperX returned no segments.")

        # ------------------------------------------------
        # STEP 2 — Build per-speaker reference clips
        #          (use the first long-enough segment for each speaker)
        # ------------------------------------------------
        print(f"[{job_id}] Building speaker reference clips …")
        src_lang_nllb = detect_language_nllb(detected_language)

        speaker_refs: dict[str, str] = {}   # speaker_id → path to reference WAV

        # Collect the longest segment per speaker for a better voice clone
        speaker_best: dict[str, dict] = {}
        for seg in segments:
            spk = seg.get("speaker", "SPEAKER_00")
            dur = seg["end"] - seg["start"]
            if spk not in speaker_best or dur > (speaker_best[spk]["end"] - speaker_best[spk]["start"]):
                speaker_best[spk] = seg

        for spk, seg in speaker_best.items():
            ref_path = os.path.join(TEMP_DIR, f"{job_id}_ref_{spk}.wav")
            extract_speaker_reference(
                original_audio_path,
                seg["start"],
                seg["end"],
                ref_path,
            )
            speaker_refs[spk] = ref_path
            print(f"[{job_id}]   Reference for {spk}: {seg['start']:.1f}s–{seg['end']:.1f}s")

        # ------------------------------------------------
        # STEP 3 — NLLB translation + XTTS synthesis
        # ------------------------------------------------
        print(f"[{job_id}] Step 2+3: Translate → TTS …")

        original_audio_pydub = AudioSegment.from_file(original_audio_path)
        total_duration_ms = len(original_audio_pydub)
        final_audio = AudioSegment.silent(duration=total_duration_ms)

        errors = []

        for idx, seg in enumerate(segments):
            spk = seg.get("speaker", "SPEAKER_00")
            original_text = seg.get("text", "").strip()
            start_sec = float(seg["start"])
            end_sec = float(seg["end"])
            duration_sec = end_sec - start_sec

            if not original_text or duration_sec <= 0:
                continue

            try:
                # Translate
                arabic_text = translate_to_arabic(original_text, src_lang_nllb)
                print(f"[{job_id}] [{idx}] {spk} | {original_text!r} → {arabic_text!r}")

                # Paths for this segment
                raw_tts_path = os.path.join(TEMP_DIR, f"{job_id}_{idx}_raw.wav")
                aligned_tts_path = os.path.join(TEMP_DIR, f"{job_id}_{idx}_aligned.wav")

                # Synthesise Arabic speech in speaker's cloned voice
                speaker_wav = speaker_refs.get(spk)
                if not speaker_wav or not os.path.exists(speaker_wav):
                    print(f"[{job_id}] [{idx}] Warning: no reference for {spk}, skipping.")
                    continue

                tts_model.tts_to_file(
                    text=arabic_text,
                    file_path=raw_tts_path,
                    speaker_wav=speaker_wav,
                    language="ar",
                )

                # Time-stretch to fit original segment duration
                stretch_audio(raw_tts_path, aligned_tts_path, duration_sec)

                # Overlay at the exact original timestamp
                generated_seg = AudioSegment.from_file(aligned_tts_path)
                final_audio = final_audio.overlay(
                    generated_seg,
                    position=int(start_sec * 1000),
                )

            except Exception as e:
                msg = f"Segment {idx} ({spk}) failed: {e}"
                print(f"[{job_id}] ERROR: {msg}")
                errors.append(msg)
                # Continue with remaining segments rather than aborting

        # ------------------------------------------------
        # STEP 4 — Export
        # ------------------------------------------------
        output_path = os.path.join(OUTPUT_DIR, f"{job_id}_dubbed.wav")
        final_audio.export(output_path, format="wav")
        print(f"[{job_id}] Done → {output_path}")

        if errors:
            print(f"[{job_id}] Completed with {len(errors)} segment error(s):")
            for e in errors:
                print(f"  • {e}")

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="dubbed.wav",
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always clean up temp files for this job
        cleanup_temp_files(job_id)


# ============================================
# HEALTH CHECK
# ============================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "whisperx_model": WHISPERX_MODEL,
        "translation_model": NLLB_MODEL_ID,
        "tts_model": "xtts_v2",
        "target_language": TARGET_LANG_NLLB,
    }
