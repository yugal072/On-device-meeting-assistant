import queue
import json
import sys
import time
import threading

import numpy as np
import sounddevice as sd
import torch

from faster_whisper import WhisperModel


# ==============================
# Audio Config
# ==============================

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 5  # seconds per chunk


# ==============================
# Audio Buffer
# ==============================

audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    """Called for each audio block"""

    if status:
        print(status, file=sys.stderr)

    audio_queue.put(indata.copy())


# ==============================
# Whisper Model
# ==============================

def load_model():

    import os
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    device = "cpu"
    compute_type = "int8"

    print("Loading Whisper model (this may take a few minutes on first run)...")

    model = WhisperModel(
        "base",          # Use base for faster download
        device=device,
        compute_type=compute_type,
        download_root="./models"   # Local folder
    )

    print("Model loaded successfully.")

    return model



# ==============================
# Audio Collector
# ==============================

def record_audio():

    print("🎙️  Listening... Press CTRL+C to stop.\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.5)
    ):
        while True:
            time.sleep(0.1)


# ==============================
# Transcription Worker
# ==============================

def transcriber(model):

    buffer = np.zeros(0, dtype=np.float32)
    chunks = []

    start_time = time.time()

    while True:

        data = audio_queue.get()

        data = data.flatten()
        buffer = np.concatenate([buffer, data])

        # When enough audio collected
        if len(buffer) >= SAMPLE_RATE * BLOCK_DURATION:

            audio_chunk = buffer.copy()
            buffer = np.zeros(0, dtype=np.float32)

            segments, _ = model.transcribe(
                audio_chunk,
                vad_filter=True
            )

            for segment in segments:

                chunk = {
                    "timestamp": round(time.time() - start_time, 2),
                    "speaker": "unknown",
                    "text": segment.text.strip()
                }

                chunks.append(chunk)

                print(f"[{chunk['timestamp']}s] {chunk['text']}")

            # Save continuously
            with open("live_transcript.json", "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)


# ==============================
# Main
# ==============================

def main():

    model = load_model()

    # Start recording thread
    record_thread = threading.Thread(target=record_audio, daemon=True)
    record_thread.start()

    # Start transcription loop
    transcriber(model)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
