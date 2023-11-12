import assemblyai as aai
import os

aai.settings.api_key = os.environ.get("assemblyai_key")

#replace with your file
FILE_URL = "https://github.com/AssemblyAI-Examples/audio-examples/raw/main/20230607_me_canadian_wildfires.mp3"
FILE_URL = "https://storage.googleapis.com/aai-web-samples/meeting.mp4"

transcriber = aai.Transcriber()

transcript = transcriber.transcribe(FILE_URL)

#take the transcript id and use it elsewhere in this repo
print(transcript.id)