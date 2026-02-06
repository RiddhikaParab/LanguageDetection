import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
from langdetect import detect
from langdetect import detect_langs
import langcodes
import time

SAMPLE_RATE = 16000
DURATION = 7  # longer chunk helps mixed language detection

print("🔄 Loading Whisper model...")
model = whisper.load_model("base")
print("✅ Model loaded")

def record_chunk(duration=DURATION):
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16
    )
    sd.wait()
    audio = audio / np.max(np.abs(audio))
    return audio

def detect_language_audio(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)

    lang_code = max(probs, key=probs.get)
    lang_name = langcodes.Language.get(lang_code).display_name()

    return lang_code, lang_name, probs[lang_code]


def process_audio(audio_chunk):
    temp_path = None

    try:
        # Save audio chunk
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            wav.write(temp_path, SAMPLE_RATE, audio_chunk)

        # 🔹 STEP 2: AUDIO-BASED DOMINANT LANGUAGE
        lang_code, lang_name, confidence = detect_language_audio(temp_path)

        if confidence < 0.35:
            print(f"\n⚠️ Low confidence ({confidence:.2f}), continuing anyway")
            print(f"🔍 Raw confidence score: {confidence:.2f}")


        print(f"\n🌍 Dominant Language: {lang_name} ({confidence:.2f})")

        # 🔹 Transcription
        result = model.transcribe(temp_path, fp16=False)

        # 🔹 STEP 3: MIXED LANGUAGE DETECTION (TEXT-BASED)
        full_text = " ".join(
            seg["text"] for seg in result["segments"]
            if len(seg["text"].strip()) >= 5
        )

        print("\n📝 Transcription:")
        print(full_text)

        try:
            langs = detect_langs(full_text)
            print("\n🌐 Possible mixed languages:")
            for l in langs:
                lang = langcodes.Language.get(l.lang).display_name()
                print(f"{lang}: {l.prob:.2f}")
        except:
            pass

    finally:
        if temp_path and os.path.exists(temp_path):
            time.sleep(0.2)
            try:
                os.remove(temp_path)
            except PermissionError:
                pass

try:
    while True:
        print("\n🎧 Listening...")
        audio = record_chunk()
        process_audio(audio)

except KeyboardInterrupt:
    print("\n🛑 Stopped")

