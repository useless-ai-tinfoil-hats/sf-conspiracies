import assemblyai as aai

aai.settings.api_key = "b22bc6fc4ce046f08fce971260e70ce6"

#FILE_URL = "https://github.com/AssemblyAI-Community/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"

# You can also transcribe a local file by passing in a file path
FILE_URL = '/Users/vitorialarasoria/Desktop/hackathon/output_audio.wav'

transcriber = aai.Transcriber()
transcript = transcriber.transcribe(FILE_URL)

if transcript.status == aai.TranscriptStatus.error:
    print(transcript.error)
else:
    print(transcript.text)
