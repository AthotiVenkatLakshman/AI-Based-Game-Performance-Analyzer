import io
import json
import os
import subprocess
import tempfile


def text_to_speech(text, lang_code="en"):
    """
    Offline TTS using macOS `say` command + ffmpeg → returns BytesIO(mp3).
    Falls back to a silent BytesIO if something goes wrong.
    """
    # Truncate very long texts to avoid slow generation
    text = text[:1500]

    # Map lang_code to macOS `say` voice
    voice_map = {
        "en": "Samantha",   # macOS English voice
        "hi": "Lekha",      # macOS Hindi voice (installed by default on most Macs)
        "te": "Samantha",   # Telugu not natively in macOS say; fall back to English
    }
    voice = voice_map.get(lang_code, "Samantha")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            aiff_path = os.path.join(tmpdir, "speech.aiff")
            mp3_path  = os.path.join(tmpdir, "speech.mp3")

            # Generate AIFF via macOS say (completely offline)
            subprocess.run(
                ["say", "-v", voice, "-o", aiff_path, text],
                check=True,
                capture_output=True,
                timeout=30
            )

            # Convert AIFF → MP3 via ffmpeg
            subprocess.run(
                ["ffmpeg", "-y", "-i", aiff_path, mp3_path],
                check=True,
                capture_output=True,
                timeout=30
            )

            with open(mp3_path, "rb") as f:
                audio_bytes = io.BytesIO(f.read())
            audio_bytes.seek(0)
            return audio_bytes

    except Exception as e:
        raise RuntimeError(f"TTS failed: {e}")


def save_chat_history(history, filename="chat_history.json"):
    """Saves session history to a local JSON file."""
    with open(filename, "w") as f:
        json.dump(history, f, indent=4)


def get_language_code(language):
    mapping = {
        "English": "en",
        "Hindi": "hi",
        "Telugu": "te"
    }
    return mapping.get(language, "en")
