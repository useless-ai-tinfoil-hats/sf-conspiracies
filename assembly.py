import pyaudio
import wave
import threading
import assemblyai as aai

aai.settings.api_key = "b22bc6fc4ce046f08fce971260e70ce6"

def record_audio(output_filename, sample_rate=44100, chunk_size=1024):
    audio_format = pyaudio.paInt16
    channels = 1
    rate = sample_rate
    chunk = chunk_size
    audio = pyaudio.PyAudio()

    stream = audio.open(format=audio_format, channels=channels,
                        rate=rate, input=True, frames_per_buffer=chunk)

    print(f"Recording... Press 'Enter' to stop.")

    frames = []

    def record():
        while not stop_recording_event.is_set():
            data = stream.read(chunk)
            frames.append(data)


    stop_recording_event = threading.Event()

    record_thread = threading.Thread(target=record)
    record_thread.start()

    input("Press 'Enter' to stop the recording.\n")
    stop_recording_event.set()
    record_thread.join()
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(audio_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio recording saved as {output_filename}")

output_file = "teste123.wav"
demo_file = "piggeons_demo.wav"
record_audio(output_file)

config = aai.TranscriptionConfig(speaker_labels=True)

transcriber = aai.Transcriber()
transcript = transcriber.transcribe(
  output_file,
  config=config
)

for utterance in transcript.utterances:
  print(f"Speaker {utterance.speaker}: {utterance.text}")

