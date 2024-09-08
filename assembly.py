import pyaudio
import wave
import threading
import assemblyai as aai


def assembly_detect_speakers(demo_file): 
  aai.settings.api_key = "b22bc6fc4ce046f08fce971260e70ce6"
  text = []


  config = aai.TranscriptionConfig(speaker_labels=True)

  transcriber = aai.Transcriber()
  transcript = transcriber.transcribe(
    demo_file,
    config=config
  )

  for utterance in transcript.utterances:
    text.append(f"Speaker {utterance.speaker}: {utterance.text}")
  
  return "\n".join(text)
