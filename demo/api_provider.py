"""
VibeVoice API Provider
======================
Abstraction layer that lets you swap the local VibeVoice model for any
OpenAI-compatible TTS API (OpenAI, Azure OpenAI, etc.).

Usage
-----
The factory function `build_tts_backend()` reads the environment (loaded from
.env by `load_dotenv`) and returns either:

  LocalTTSBackend   – loads weights from disk, original behaviour
  APITTSBackend     – calls an OpenAI-compatible /audio/speech endpoint

Both backends expose the same interface:

    backend.generate(script, speaker_names, cfg_scale, ...) -> np.ndarray
"""

from __future__ import annotations

import io
import os
import re
import traceback
from typing import List, Optional, Tuple

import numpy as np

# Optional heavy imports – only pulled in when actually needed
_librosa = None
_sf = None


def _lazy_audio_imports():
    global _librosa, _sf
    if _librosa is None:
        import librosa as _lib
        import soundfile as _snd
        _librosa = _lib
        _sf = _snd


# ─── Config helpers ───────────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _parse_voices(raw: str) -> List[str]:
    """Parse a comma-separated voice list ("A, B, C") into trimmed strings."""
    return [v.strip() for v in raw.split(",") if v.strip()]


def get_mode() -> str:
    """Return 'local' or 'api' based on VIBEVOICE_MODE env var."""
    return _env("VIBEVOICE_MODE", "local").lower()


def get_model_path() -> str:
    return _env("MODEL_PATH", "vibevoice/VibeVoice-1.5B")


# ─── Base class ───────────────────────────────────────────────────────────────

class TTSBackend:
    """Minimal interface every backend must implement."""

    @property
    def info(self) -> str:
        return "TTS Backend"

    def generate(
        self,
        script: str,
        voice_samples: Optional[List[np.ndarray]] = None,
        cfg_scale: float = 1.3,
        inference_steps: int = 10,
        seed: Optional[int] = None,
        voice_cloning_enabled: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from a formatted script.

        Returns
        -------
        (audio_array, sample_rate)
            audio_array : float32 numpy array, shape (N,)
            sample_rate : int  (typically 24 000)
        """
        raise NotImplementedError

    @staticmethod
    def parse_script_segments(script: str) -> List[Tuple[int, str]]:
        """
        Parse a script into (speaker_order_idx, text) segments.
        Expected format: "Speaker 1: text\nSpeaker 2: text\n...".

        Speaker numbers are normalized to 0-based ORDER OF FIRST APPEARANCE,
        so "Speaker 0"/"Speaker 1" and "Speaker 1"/"Speaker 2" both map to
        indices 0, 1, ... regardless of how the user numbered them.

        Returns list of (speaker_order_idx, text).
        """
        segments = []
        for line in script.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^Speaker\s+(\d+)\s*:\s*(.*)$", line, re.IGNORECASE)
            if m:
                speaker_num = int(m.group(1))
                text = m.group(2).strip()
                if text:
                    segments.append((speaker_num, text))

        # Normalize by order of first appearance → 0, 1, 2, ...
        order_map = {}
        normalized = []
        for speaker_num, text in segments:
            if speaker_num not in order_map:
                order_map[speaker_num] = len(order_map)
            normalized.append((order_map[speaker_num], text))
        return normalized

    @staticmethod
    def stitch_audio(segments: List[Tuple[np.ndarray, int]], pause_ms: int = 300) -> Tuple[np.ndarray, int]:
        """
        Stitch multiple audio segments with a short pause between speakers.
        segments: list of (audio_np, sample_rate) — all must have same sample_rate.
        Returns (stitched_audio, sample_rate).
        """
        if not segments:
            return np.array([], dtype=np.float32), 24000

        # Verify all have same sample rate
        sr = segments[0][1]
        for _, s in segments:
            if s != sr:
                raise ValueError("All segments must have the same sample rate")

        pause_samples = int(sr * pause_ms / 1000)
        pause = np.zeros(pause_samples, dtype=np.float32)

        parts = []
        for i, (audio, _) in enumerate(segments):
            if i > 0:
                parts.append(pause)
            parts.append(audio)

        return np.concatenate(parts), sr


# ─── Local backend (original behaviour) ──────────────────────────────────────

class LocalTTSBackend(TTSBackend):
    """
    Loads VibeVoice weights locally and runs inference on this machine.
    This is the original behaviour; nothing changed here.
    """

    @property
    def info(self) -> str:
        return f"Local VibeVoice ({self.device})"

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        inference_steps: int = 10,
        adapter_path: Optional[str] = None,
    ):
        import torch
        import traceback as _tb

        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.modular.lora_loading import load_lora_assets
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        self.device = device
        self.inference_steps = inference_steps
        self.sample_rate = 24_000

        print(f"[LocalTTSBackend] Loading model from {model_path}")
        self.processor = VibeVoiceProcessor.from_pretrained(model_path)

        if device == "mps":
            load_dtype = torch.float16
            attn = "sdpa"
        elif device == "cuda":
            load_dtype = torch.bfloat16
            attn = "flash_attention_2"
        else:
            load_dtype = torch.float32
            attn = "sdpa"

        try:
            if device == "mps":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path, torch_dtype=load_dtype, attn_implementation=attn,
                    device_map=None,
                )
                self.model.to("mps")
            else:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path, torch_dtype=load_dtype,
                    device_map=device if device in ("cuda", "cpu") else "cpu",
                    attn_implementation=attn,
                )
        except Exception as exc:
            if attn == "flash_attention_2":
                print(f"[LocalTTSBackend] flash_attention_2 failed, falling back to sdpa: {exc}")
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path, torch_dtype=load_dtype,
                    device_map=device if device in ("cuda", "cpu") else "cpu",
                    attn_implementation="sdpa",
                )
                if device == "mps":
                    self.model.to("mps")
            else:
                raise

        if adapter_path:
            print(f"[LocalTTSBackend] Loading LoRA from {adapter_path}")
            load_lora_assets(self.model, adapter_path)

        self.model.eval()
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=inference_steps)

    def generate(
        self,
        script: str,
        voice_samples: Optional[List[np.ndarray]] = None,
        cfg_scale: float = 1.3,
        inference_steps: int = 10,
        seed: Optional[int] = None,
        voice_cloning_enabled: bool = True,
    ) -> Tuple[np.ndarray, int]:
        import torch

        self.model.set_ddpm_inference_steps(num_steps=inference_steps)

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda" if self.device == "cuda" else "cpu")
            generator.manual_seed(seed)

        inputs = self.processor(
            text=[script],
            voice_samples=[voice_samples] if voice_samples else None,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        target_device = self.device if self.device in ("cuda", "mps") else "cpu"
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(target_device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            generator=generator,
            verbose=False,
            is_prefill=voice_cloning_enabled,
        )

        audio = outputs.speech_outputs[0]
        if torch.is_tensor(audio):
            if audio.dtype == torch.bfloat16:
                audio = audio.float()
            audio = audio.cpu().numpy()

        return audio.astype(np.float32), self.sample_rate


# ─── Gemini TTS backend (Google AI Studio) ───────────────────────────────────

class GeminiTTSBackend(TTSBackend):
    """
    Calls the Gemini API (Google AI Studio) for text-to-speech.
    Uses the generateContent endpoint with response_modalities=["AUDIO"].

    Required .env keys:
        API_KEY    – your Google AI Studio API key (starts with AQ. or AIza...)
        API_MODEL  – Gemini model, e.g. gemini-2.5-flash  (default)
                     Other options: gemini-2.5-pro
        API_VOICE  – voice name, e.g. Aoede | Charon | Fenrir | Kore | Puck
                     Full list: https://ai.google.dev/gemini-api/docs/speech-generation

    Optional:
        API_TIMEOUT – request timeout in seconds
    """

    _BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    # Available Gemini TTS voices (as of 2026)
    VOICES = ["Aoede", "Charon", "Fenrir", "Kore", "Puck",
              "Achernar", "Achird", "Algenib", "Algieba", "Alnilam"]

    @property
    def info(self) -> str:
        return f"Gemini ({self.model}, voice={self.voice})"

    def __init__(self):
        self.api_key = _env("API_KEY")
        self.base_url = self._BASE
        self.model = _env("API_MODEL", "gemini-2.5-flash-preview-tts")
        self.voice = _env("API_VOICE", "Kore")
        self.voices = _parse_voices(_env("API_VOICES")) or [self.voice]
        self.timeout = int(_env("API_TIMEOUT", "120"))
        self.sample_rate = 24_000

        if not self.api_key:
            raise ValueError("API_KEY is not set in .env.")

        print(f"[GeminiTTSBackend] model={self.model}  voice={self.voice}  voices={self.voices[:4]}{'…' if len(self.voices) > 4 else ''}")

    @staticmethod
    def _strip_speaker_labels(script: str) -> str:
        lines = []
        for line in script.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^Speaker\s+\d+:\s*", "", line, flags=re.IGNORECASE)
            lines.append(line)
        return " ".join(lines)

    def generate(
        self,
        script: str,
        voice_samples=None,
        cfg_scale: float = 1.3,
        inference_steps: int = 10,
        seed=None,
        voice_cloning_enabled: bool = True,
        voice: Optional[str] = None,
    ):
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required. Install: pip install httpx")

        _lazy_audio_imports()

        text = self._strip_speaker_labels(script)
        if not text:
            raise ValueError("Script is empty after stripping speaker labels.")

        voice_name = voice if voice is not None else self.voice

        url = f"{self._BASE}/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": voice_name
                        }
                    }
                }
            }
        }

        print(f"[GeminiTTSBackend] POST {self._BASE}/{self.model}:generateContent  chars={len(text)}")

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini API error [{response.status_code}]: {response.text[:500]}"
            )

        import base64
        data = response.json()

        # Extract audio from response
        try:
            parts = data["candidates"][0]["content"]["parts"]
            audio_b64 = None
            for part in parts:
                if "inlineData" in part:
                    audio_b64 = part["inlineData"]["data"]
                    mime = part["inlineData"].get("mimeType", "audio/wav")
                    break
            if audio_b64 is None:
                raise ValueError(f"No audio in response. Full response: {str(data)[:300]}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Gemini response structure: {e}\n{str(data)[:300]}")

        audio_bytes = base64.b64decode(audio_b64)
        audio_io = io.BytesIO(audio_bytes)

        # Gemini returns PCM or WAV depending on model
        try:
            audio_np, sr = _sf.read(audio_io, dtype="float32")
        except Exception:
            # Fallback: raw 16-bit PCM at 24kHz
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sr = 24_000

        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        if sr != 24_000:
            audio_np = _librosa.resample(audio_np, orig_sr=sr, target_sr=24_000)
            sr = 24_000

        self.sample_rate = sr
        print(f"[GeminiTTSBackend] Received {len(audio_np)/sr:.1f}s at {sr} Hz")
        return audio_np.astype(np.float32), sr


# ─── Google Cloud TTS backend ─────────────────────────────────────────────────

class GoogleTTSBackend(TTSBackend):
    """
    Calls the Google Cloud Text-to-Speech REST API.

    Required .env keys:
        API_KEY        – your Google Cloud API key (or use GOOGLE_APPLICATION_CREDENTIALS)
        API_MODEL      – voice name, e.g. en-US-Neural2-D, en-US-Wavenet-F
                         Full list: https://cloud.google.com/text-to-speech/docs/voices
        API_VOICE      – speaking rate as a float string, e.g. "1.0" (default)

    Optional:
        API_AUDIO_FORMAT  – LINEAR16 | MP3 | OGG_OPUS  (default LINEAR16 → wav)
        API_TIMEOUT       – request timeout in seconds
    """

    _ENDPOINT = "https://texttospeech.googleapis.com/v1/text:synthesize"

    @property
    def info(self) -> str:
        return f"Google Cloud TTS ({self.voice_name})"

    def __init__(self):
        self.api_key = _env("API_KEY")
        self.base_url = self._ENDPOINT
        self.model = _env("API_MODEL", "en-US-Neural2-D")
        self.voice_name = self.model
        # API_VOICE is a speaking RATE in this provider; per-speaker voices come
        # from API_VOICES (a comma-separated list of Cloud TTS voice names).
        self.voices = _parse_voices(_env("API_VOICES"))
        self.speaking_rate = float(_env("API_VOICE", "1.0") or "1.0")
        self.voice = self.speaking_rate  # keep info property happy
        self.audio_encoding = _env("API_AUDIO_FORMAT", "LINEAR16").upper()
        self.timeout = int(_env("API_TIMEOUT", "60"))
        self.sample_rate = 24_000

        if not self.api_key or self.api_key.startswith("sk-..."):
            raise ValueError(
                "API_KEY is not set. Provide your Google Cloud API key in .env."
            )

        # Derive language code from voice name prefix (e.g. "en-US-Neural2-D" → "en-US")
        parts = self.voice_name.split("-")
        self.language_code = "-".join(parts[:2]) if len(parts) >= 2 else "en-US"

        print(
            f"[GoogleTTSBackend] voice={self.voice_name}  lang={self.language_code}  "
            f"rate={self.speaking_rate}  encoding={self.audio_encoding}  "
            f"voices={self.voices[:4]}{'…' if len(self.voices) > 4 else ''}"
        )

    @staticmethod
    def _strip_speaker_labels(script: str) -> str:
        lines = []
        for line in script.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^Speaker\s+\d+:\s*", "", line, flags=re.IGNORECASE)
            lines.append(line)
        return " ".join(lines)

    def generate(
        self,
        script: str,
        voice_samples=None,
        cfg_scale: float = 1.3,
        inference_steps: int = 10,
        seed=None,
        voice_cloning_enabled: bool = True,
        voice: Optional[str] = None,
    ):
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for API mode. Install: pip install httpx")

        _lazy_audio_imports()

        text = self._strip_speaker_labels(script)
        if not text:
            raise ValueError("Script is empty after stripping speaker labels.")

        # Per-speaker voice name (e.g. "en-US-Neural2-F"); falls back to API_MODEL.
        voice_name = voice if voice is not None else self.voice_name
        lang_code = self.language_code
        if voice is not None:
            vparts = voice_name.split("-")
            lang_code = "-".join(vparts[:2]) if len(vparts) >= 2 else self.language_code

        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": lang_code,
                "name": voice_name,
            },
            "audioConfig": {
                "audioEncoding": self.audio_encoding,
                "speakingRate": self.speaking_rate,
                "sampleRateHertz": 24000,
            },
        }
        url = f"{self._ENDPOINT}?key={self.api_key}"
        headers = {"Content-Type": "application/json"}

        print(f"[GoogleTTSBackend] POST {self._ENDPOINT}  voice={voice_name}  chars={len(text)}")

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"Google TTS API error [{response.status_code}]: {response.text[:500]}"
            )

        import base64
        data = response.json()
        audio_bytes = base64.b64decode(data["audioContent"])

        audio_io = io.BytesIO(audio_bytes)
        audio_np, sr = _sf.read(audio_io, dtype="float32")

        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        if sr != 24_000:
            audio_np = _librosa.resample(audio_np, orig_sr=sr, target_sr=24_000)
            sr = 24_000

        self.sample_rate = sr
        print(f"[GoogleTTSBackend] Received {len(audio_np)/sr:.1f}s at {sr} Hz")
        return audio_np.astype(np.float32), sr


# ─── ElevenLabs TTS backend ───────────────────────────────────────────────────

class ElevenLabsTTSBackend(TTSBackend):
    """
    Calls the ElevenLabs Text-to-Speech API.

    Required .env keys:
        API_KEY    – your ElevenLabs API key (xi-api-key)
        API_MODEL  – voice ID (not name), e.g. 21m00Tcm4TlvDq8ikWAM
                     Find yours at https://api.elevenlabs.io/v1/voices

    Optional:
        API_VOICE           – model ID string, e.g. eleven_multilingual_v2  (default)
        API_AUDIO_FORMAT    – mp3_44100_128 | pcm_24000 | ulaw_8000  (default pcm_24000)
        API_TIMEOUT         – request timeout in seconds
    """

    _BASE = "https://api.elevenlabs.io/v1"

    @property
    def info(self) -> str:
        return f"ElevenLabs ({self.model_id}, voice={self.voice_id})"

    def __init__(self):
        self.api_key = _env("API_KEY")
        self.base_url = self._BASE
        self.voice_id = _env("API_MODEL", "")
        self.model = self.voice_id
        self.model_id = _env("API_VOICE", "eleven_multilingual_v2")
        # Multi-speaker: comma-separated voice IDs, one per speaker.
        # Each entry is a voice ID (like API_MODEL).
        self.voices = _parse_voices(_env("API_VOICES")) or []
        self.voice = self.voice_id  # info property support
        self.output_format = _env("API_AUDIO_FORMAT", "pcm_24000")
        self.timeout = int(_env("API_TIMEOUT", "60"))
        self.sample_rate = 24_000

        if not self.api_key or self.api_key.startswith("sk-..."):
            raise ValueError(
                "API_KEY is not set. Provide your ElevenLabs API key in .env."
            )
        if not self.voice_id and not self.voices:
            raise ValueError(
                "API_MODEL must be set to a valid ElevenLabs voice ID (or set API_VOICES).\n"
                "Find yours at https://api.elevenlabs.io/v1/voices using your API key."
            )

        print(
            f"[ElevenLabsTTSBackend] voice_id={self.voice_id}  model={self.model_id}  "
            f"format={self.output_format}  voices={self.voices[:4]}{'…' if len(self.voices) > 4 else ''}"
        )

    @staticmethod
    def _strip_speaker_labels(script: str) -> str:
        lines = []
        for line in script.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^Speaker\s+\d+:\s*", "", line, flags=re.IGNORECASE)
            lines.append(line)
        return " ".join(lines)

    def generate(
        self,
        script: str,
        voice_samples=None,
        cfg_scale: float = 1.3,
        inference_steps: int = 10,
        seed=None,
        voice_cloning_enabled: bool = True,
        voice: Optional[str] = None,
    ):
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx is required for API mode. Install: pip install httpx")

        _lazy_audio_imports()

        text = self._strip_speaker_labels(script)
        if not text:
            raise ValueError("Script is empty after stripping speaker labels.")

        voice_id = voice if voice is not None else self.voice_id

        url = f"{self._BASE}/text-to-speech/{voice_id}?output_format={self.output_format}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self.model_id,
        }

        print(f"[ElevenLabsTTSBackend] POST {url}  chars={len(text)}")

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"ElevenLabs API error [{response.status_code}]: {response.text[:500]}"
            )

        # pcm_24000 → raw 16-bit PCM; everything else → soundfile-readable
        if self.output_format.startswith("pcm_"):
            sr = int(self.output_format.split("_")[1])
            audio_np = np.frombuffer(response.content, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_io = io.BytesIO(response.content)
            audio_np, sr = _sf.read(audio_io, dtype="float32")
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)

        if sr != 24_000:
            audio_np = _librosa.resample(audio_np, orig_sr=sr, target_sr=24_000)
            sr = 24_000

        self.sample_rate = sr
        print(f"[ElevenLabsTTSBackend] Received {len(audio_np)/sr:.1f}s at {sr} Hz")
        return audio_np.astype(np.float32), sr


# ─── OpenAI-compatible API backend ────────────────────────────────────────────

class APITTSBackend(TTSBackend):
    """
    Calls an OpenAI-compatible /audio/speech endpoint.

    The script text is passed as the 'input' field.  Speaker formatting and
    voice-cloning are handled upstream (in the demo/inference scripts); this
    backend just fires the HTTP request and decodes the audio response.

    Supported providers (anything with an OpenAI-compatible API):
      • OpenAI           – https://api.openai.com/v1
      • Azure OpenAI     – https://RESOURCE.openai.azure.com/openai/deployments/DEPLOY
      • Any compatible   – set API_BASE_URL to their base URL
    """

    @property
    def info(self) -> str:
        return f"OpenAI-compatible ({self.model} @ {self.base_url}, voice={self.voice})"

    def __init__(self):
        self.api_key = _env("API_KEY")
        self.base_url = _env("API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = _env("API_MODEL", "tts-1")
        self.voice = _env("API_VOICE", "alloy")
        # Multi-speaker: comma-separated voices, one per speaker.
        # E.g. API_VOICES=alloy,echo,fable,onyx
        self.voices = _parse_voices(_env("API_VOICES")) or []
        self.audio_format = _env("API_AUDIO_FORMAT", "wav")
        self.timeout = int(_env("API_TIMEOUT", "60"))
        self.sample_rate = 24_000  # default; overridden after decoding

        if not self.api_key or self.api_key.startswith("sk-..."):
            raise ValueError(
                "API_KEY is not set in .env. "
                "Set VIBEVOICE_MODE=local to use the local model instead."
            )

        print(
            f"[APITTSBackend] provider={self.base_url}  model={self.model}  "
            f"voice={self.voice}  voices={self.voices[:4]}{'…' if len(self.voices) > 4 else ''}  "
            f"format={self.audio_format}"
        )

    def _strip_script_to_text(self, script: str) -> str:
        """
        Strip 'Speaker N:' prefixes so the API receives clean dialogue text.
        Multiple lines are joined with a space.
        """
        lines = []
        for line in script.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Remove 'Speaker N: ' prefix
            line = re.sub(r"^Speaker\s+\d+:\s*", "", line, flags=re.IGNORECASE)
            lines.append(line)
        return " ".join(lines)

    def generate(
        self,
        script: str,
        voice_samples: Optional[List[np.ndarray]] = None,  # ignored by API
        cfg_scale: float = 1.3,                            # ignored by API
        inference_steps: int = 10,                          # ignored by API
        seed: Optional[int] = None,                         # ignored by API
        voice_cloning_enabled: bool = True,                 # ignored by API
        voice: Optional[str] = None,                        # per-speaker voice
    ) -> Tuple[np.ndarray, int]:
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for API mode. Install it with: pip install httpx"
            )

        _lazy_audio_imports()

        text_input = self._strip_script_to_text(script)
        if not text_input:
            raise ValueError("Script produced empty text after stripping speaker labels.")

        voice_name = voice if voice is not None else self.voice

        url = f"{self.base_url}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": text_input,
            "voice": voice_name,
            "response_format": self.audio_format,
        }

        print(f"[APITTSBackend] POST {url}  voice={voice_name}  input_length={len(text_input)} chars")

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"API request failed [{response.status_code}]: {response.text[:500]}"
            )

        # Decode audio bytes → numpy float32
        audio_bytes = io.BytesIO(response.content)
        audio_np, sr = _sf.read(audio_bytes, dtype="float32")

        # Flatten stereo → mono
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        # Resample to 24 kHz if needed
        if sr != 24_000:
            audio_np = _librosa.resample(audio_np, orig_sr=sr, target_sr=24_000)
            sr = 24_000

        self.sample_rate = sr
        print(f"[APITTSBackend] Received {len(audio_np)/sr:.1f}s of audio at {sr} Hz")
        return audio_np, sr


# ─── Multi-speaker orchestration ─────────────────────────────────────────────

def generate_multi_speaker(
    backend: TTSBackend,
    script: str,
    num_speakers: int = 0,
    pause_ms: int = 300,
) -> Tuple[np.ndarray, int]:
    """
    Generate a multi-speaker podcast from a "Speaker N:" script.

    Each speaker segment is generated with its OWN voice (pulled from the
    backend's `voices` list) and stitched together with a short pause — this
    is what makes Speaker 1 / Speaker 2 / ... actually sound different in
    API mode.

    If the script has no "Speaker N:" labels, the whole thing is generated
    with the backend's default voice (single speaker).

    Parameters
    ----------
    backend      : any API backend (must expose .voices / .voice and .generate(voice=...))
    script       : "Speaker 1: ...\nSpeaker 2: ...\n..."
    num_speakers : (optional) expected speaker count — used only for logging
    pause_ms     : silence gap between speakers, in milliseconds

    Returns
    -------
    (stitched_audio, sample_rate)
    """
    segments = backend.parse_script_segments(script)

    # No speaker labels → single-voice pass-through
    if not segments:
        print("[generate_multi_speaker] No 'Speaker N:' labels -> single voice")
        return backend.generate(script=script)

    voices = getattr(backend, "voices", None) or []
    default_voice = getattr(backend, "voice", None)

    if not voices:
        print(
            f"[generate_multi_speaker] {len(segments)} speaker segment(s), "
            f"{num_speakers or 'unknown'} speaker(s) requested — no API_VOICES set, "
            f"all speakers will use the default voice ({default_voice!r}). "
            f"Set API_VOICES to give each speaker a different voice."
        )
        # Every speaker uses the single configured voice
        stitched = []
        for _, text in segments:
            stitched.append(backend.generate(script=text, voice=default_voice))
        return TTSBackend.stitch_audio(stitched, pause_ms=pause_ms)

    # Cycle the voice list per speaker index: each speaker keeps ONE consistent
    # voice across all of their turns, and if fewer voices than speakers are
    # listed the mapping wraps (Speaker 1/3/5 -> voices[0], 2/4/6 -> voices[1]).
    stitched = []
    for speaker_idx, text in segments:
        voice = voices[speaker_idx % len(voices)]
        print(f"[generate_multi_speaker] Speaker {speaker_idx + 1} -> voice={voice!r}")
        stitched.append(backend.generate(script=text, voice=voice))

    return TTSBackend.stitch_audio(stitched, pause_ms=pause_ms)


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_api_provider() -> str:
    """Return the API provider name: openai | google | elevenlabs."""
    return _env("API_PROVIDER", "openai").lower()


def build_tts_backend(
    device: str = "cpu",
    inference_steps: int = 10,
    adapter_path: Optional[str] = None,
) -> TTSBackend:
    """
    Read VIBEVOICE_MODE (and API_PROVIDER) from the environment and return the
    right backend.

    VIBEVOICE_MODE=local  → LocalTTSBackend  (loads weights from MODEL_PATH)
    VIBEVOICE_MODE=api    → one of:
        API_PROVIDER=openai      → APITTSBackend      (OpenAI / Azure / compatible)
        API_PROVIDER=google      → GoogleTTSBackend   (Google Cloud TTS)
        API_PROVIDER=elevenlabs  → ElevenLabsTTSBackend

    Call `load_dotenv()` before this function if you want .env values.
    """
    mode = get_mode()
    if mode == "api":
        provider = get_api_provider()
        print(f"[build_tts_backend] mode=api  provider={provider}")
        if provider == "google":
            return GoogleTTSBackend()
        elif provider == "elevenlabs":
            return ElevenLabsTTSBackend()
        elif provider == "gemini":
            return GeminiTTSBackend()
        else:
            # Default: OpenAI-compatible
            return APITTSBackend()
    else:
        model_path = get_model_path()
        print(f"[build_tts_backend] mode=local -> LocalTTSBackend ({model_path})")
        return LocalTTSBackend(
            model_path=model_path,
            device=device,
            inference_steps=inference_steps,
            adapter_path=adapter_path,
        )
